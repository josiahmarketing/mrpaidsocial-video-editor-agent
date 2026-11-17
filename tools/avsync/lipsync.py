# lipsync.py <video> [start dur] — how far is the voice from the lips, measured on the finished file?
# Tracks the mouth opening on every frame (MediaPipe face landmarks), takes the loudness envelope of the file's own audio
# and reports the lag at which the two line up best. Speech is not perfectly aligned with mouth opening, so the absolute
# number is only meaningful NEXT TO THE SAME MEASUREMENT ON THE CAMERA ORIGINAL: run it on a stretch of the source take
# and on the edit, and compare. A difference of more than about one frame (33 ms) is a sync error the viewer can see.
#   MODEL=<face_landmarker.task>  WIN=<seconds per window, default 10>
# Prints one line per window and the median. Positive = the voice comes AFTER the lips; negative = voice early.
import sys, os, subprocess, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
src = sys.argv[1]; ss, du = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else (None, None)
cut = (["-ss", ss, "-t", du] if ss else []); FPS, W, H = 30, 640, 360
pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:stream_side_data=rotation", "-of", "csv=p=0", src], capture_output=True, text=True).stdout.replace("\n", ",").split(",")
w0, h0 = int(pr[0]), int(pr[1]); rot = any(x.strip() in ("90", "-90", "270") for x in pr[2:]); w0, h0 = (h0, w0) if rot else (w0, h0); H = int(round(W * h0 / w0 / 2) * 2)
lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=os.environ.get("MODEL", "face_landmarker.task")), num_faces=1))
p = subprocess.Popen(["ffmpeg", "-v", "error", *cut, "-i", src, "-vf", f"fps={FPS},scale={W}:{H}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
m = []
while True:
    b = p.stdout.read(W * H * 3)
    if len(b) < W * H * 3: break
    r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.frombuffer(b, np.uint8).reshape(H, W, 3)))
    if r.face_landmarks:
        L = r.face_landmarks[0]; d = lambda a, c: ((L[a].x - L[c].x) ** 2 * W * W + (L[a].y - L[c].y) ** 2 * H * H) ** 0.5
        m.append(d(13, 14) / max(1e-6, d(33, 263)))
    else: m.append(np.nan)
m = np.array(m); ok = ~np.isnan(m); print(f"{len(m)} frames, face in {ok.mean() * 100:.0f} %")
if ok.sum() < 60: sys.exit("no face to measure")
m = np.interp(np.arange(len(m)), np.where(ok)[0], m[ok])
a = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", *cut, "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-af", "highpass=f=120,lowpass=f=4000", "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
env = np.sqrt(np.convolve(a ** 2, np.ones(320) / 320, "same")[::160]); env = np.log10(env + 1e-4)          # 100 Hz
mo = np.interp(np.arange(len(env)) / 100, (np.arange(len(m)) + 0.5) / FPS, m)                                # a frame is on screen for 33 ms: use its middle
hp = lambda x: (x - np.convolve(x, np.ones(100) / 100, "same"))
win = int(float(os.environ.get("WIN", 10)) * 100); res = []
for s in range(0, len(env) - win + 1, win):
    x, y = hp(mo)[s:s + win], hp(env)[s:s + win]; x = (x - x.mean()) / (x.std() + 1e-9); y = (y - y.mean()) / (y.std() + 1e-9)
    lags = range(-45, 46); c = [np.mean(x[max(0, -k):win - max(0, k)] * y[max(0, k):win - max(0, -k)]) for k in lags]; j = int(np.argmax(c))
    if 0 < j < len(c) - 1: den = c[j - 1] - 2 * c[j] + c[j + 1]; off = 0.5 * (c[j - 1] - c[j + 1]) / den if den else 0
    else: off = 0
    res.append(((lags[j] + off) * 10, c[j])); print(f"  {s / 100 + float(ss or 0):7.1f} s  voice {res[-1][0]:+6.0f} ms after lips  (match {c[j]:.2f})")
good = [r for r, c in res if c > 0.25]
print(f"median {np.median(good):+.0f} ms over {len(good)} of {len(res)} windows" if good else "no window matched well enough")
