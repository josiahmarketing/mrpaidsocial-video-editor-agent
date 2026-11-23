# hairmask.py <face-locked video> <out.png> — where the hair and face of a face-locked speaker usually are.
# The class segmenter is run over the video and each pixel gets the share of frames in which it was hair or face; that
# share is thresholded (FREQ, default 0.2) and widened (PAD px, default 12). Feed the picture to cutout.py as KEEP=:
# a chair back or headrest that both models take for hair in SOME frames is cut back to where the hair is in MOST frames.
# Build it from the longest stretch of the take (the speaker moves, the mistake does not follow him) and use it on all of it.
# Env: SEG (selfie_multiclass.tflite, required), SEG_CROP="y0,y1" (rows the segmenter sees), EVERY (sample every Nth frame, 2),
#      FREQ, PAD. Also writes <out>.freq.png (the raw share, for choosing FREQ by eye).
import sys, os, subprocess, numpy as np, cv2, mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
inp, out = sys.argv[1:3]; W, H = 1080, 1920
seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(base_options=mpp.BaseOptions(model_asset_path=os.environ["SEG"]), output_confidence_masks=True, output_category_mask=False))
sy0, sy1 = map(int, os.environ.get("SEG_CROP", f"0,{H}").split(",")); EVERY = int(os.environ.get("EVERY", 2)); FREQ = float(os.environ.get("FREQ", 0.2)); PAD = int(os.environ.get("PAD", 12))
p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", inp, "-vf", f"scale={W}:{H}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
acc = np.zeros((sy1 - sy0, W), np.float32); n = 0; k = 0
while True:
    b = p.stdout.read(W * H * 3)
    if len(b) < W * H * 3: break
    if k % EVERY == 0:
        cm = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.frombuffer(b, np.uint8).reshape(H, W, 3)[sy0:sy1]))).confidence_masks
        acc += cv2.resize(((np.squeeze(cm[1].numpy_view()) + np.squeeze(cm[3].numpy_view())) > 0.5).astype(np.float32), (W, sy1 - sy0)); n += 1
    k += 1
f = np.zeros((H, W), np.float32); f[sy0:sy1] = acc / max(1, n)
cv2.imwrite(out + ".freq.png", (f * 255).astype(np.uint8))
m = cv2.dilate((f >= FREQ).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * PAD + 1, 2 * PAD + 1))) * 255
cv2.imwrite(out, m); print(f"{n} frames sampled -> {out}", file=sys.stderr)
os._exit(0)   # the segmenter's destructor raises at interpreter shutdown
