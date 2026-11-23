# cutout.py in.mp4 out_prefix — "sticker cutout" of a talking head for HTML compositions, as TWO plain videos that
# composite with CSS blend modes (no alpha channel needed, so any renderer that paints <video> frames can use them):
#   <prefix>-mk.mp4  multiply layer: white outside the sticker, black inside, with a soft drop shadow around it
#   <prefix>-fg.mp4  add layer: the speaker, premultiplied on black, plus a white sticker outline
# Stack them as SIBLINGS of the background graphics (same stacking context):
#   <video class="mk" style="mix-blend-mode:multiply">   then   <video class="fg" style="mix-blend-mode:plus-lighter">
# result = background x (1 - sticker) + speaker. Transform both videos with the same tween; never wrap them in a
# transformed/faded container (that isolates the group and the blend sees no backdrop).
# Env: MODNET (model path), OUTLINE px (default 12; 0 = none), SHADOW 0..1 (default 0.32), EVERY (matte every Nth frame),
#   MATTE="WxH" matte size in multiples of 32 (default 384x672; 512x896 is cleaner around hair and about half the speed).
#   SEG=<selfie_multiclass.tflite>  second opinion from a class segmenter (hair / skin / clothes). The matte is kept only
#       near those classes (SEG_PAD px of slack, default 14), which drops a chair back or wall art that touches the hair,
#       and clothes or skin the matte lost (a pale top against a pale wall) are filled back in below BODY_FROM (px row,
#       default 55 % of the height). SEG_CROP="y0,y1" runs the segmenter on those rows only (finer edges on a bust).
#       NECK=<px row>: above it clothes do not count, only hair and skin. A dark chair back or sofa behind the head is
#       taken for hair by the matte and for clothing by the segmenter; the segmenter's hair class stays on the hair.
#   HEAD="cx,cy,rx,ry"  for a face-locked speaker: above HEAD_BELOW (px row, default cy + ry) keep only what lies inside
#       an arch (the top half of this ellipse, straight sides below its centre) or is skin, so raised hands stay.
#       Removes chair wings beside the hair that both models take for hair. Size it to the widest the hair gets.
#   KEEP=<mask.png>  the same, with a picture instead of an arch (white = may be foreground above HEAD_BELOW);
#       hairmask.py makes one from where the hair is in most frames, which follows the real outline.
#   COOL="thr[,to_row[,dark]]"  for a dark chair back or headrest behind dark hair, which neither model can tell from the
#       hair, so pieces of it come and go from frame to frame. Dark hair is warm (red above blue), black plastic and
#       mesh are neutral to blue. Above to_row (default 60 % of the height), pixels darker than `dark` (115) whose local
#       red-minus-blue is under thr (try -4; print it for a hair patch and a chair patch first) are background.
#       COOL_SKIP="x0,y0,x1,y1" exempts a box (a black T-shirt under the chin is dark and cool too).
#   SOLID=<gamma 0..1>  inside the sticker, raise partial alpha to this power (default 1 = off). Fast hands are half
#       transparent over a wide blur and the white sticker paper shows through them; 0.55 keeps them flesh-coloured.
# The multiply layer must be EXACTLY white outside the sticker, or the whole video box dims the backdrop by a few
# per cent and shows as a rectangle the moment the layer is moved. An 8-bit RGB to YUV round trip loses 1-2 % of white
# each time, so the matte's top end is stored above nominal white here (WHITE_LIFT below). Re-apply it after any later
# re-encode of the matte (ffmpeg -vf "format=yuv420p,lutyuv=y='...'"), and prove it in the composition: shift the
# layer, take a snapshot, and compare the backdrop's colour inside and outside the video's box.
import sys, os, subprocess, time, numpy as np, cv2, onnxruntime as ort
inp, pre = sys.argv[1:3]
W, H, FPS = 1080, 1920, 30
OUT = int(os.environ.get("OUTLINE", 12)); SH = float(os.environ.get("SHADOW", 0.32)); EVERY = int(os.environ.get("EVERY", 1))
mw, mh = map(int, os.environ.get("MATTE", "384x672").split("x"))
so = ort.SessionOptions(); so.intra_op_num_threads = int(os.environ.get("THREADS", 4))
sess = ort.InferenceSession(os.environ.get("MODNET", os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "modnet.onnx")), so)
seg = None
if os.environ.get("SEG"):
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(base_options=mpp.BaseOptions(model_asset_path=os.environ["SEG"]), output_confidence_masks=True, output_category_mask=False))
    PAD = int(os.environ.get("SEG_PAD", 14)); BODY = float(os.environ.get("BODY_FROM", 0.55 * H)); sy0, sy1 = map(int, os.environ.get("SEG_CROP", f"0,{H}").split(","))
    pker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * PAD + 1, 2 * PAD + 1)); bramp = np.clip((np.arange(H, dtype=np.float32)[:, None] - (BODY - 60)) / 60, 0, 1)
    NECK = float(os.environ.get("NECK", 0)); nramp = np.clip((np.arange(H, dtype=np.float32)[:, None] - (NECK - 30)) / 60, 0, 1) if NECK else None
SOLID = float(os.environ.get("SOLID", 1))
COOL = None
if os.environ.get("COOL"):   # "thr[,to_row[,dark]]"
    v = [float(x) for x in os.environ["COOL"].split(",")]; COOL = (v[0], v[1] if len(v) > 1 else 0.6 * H, v[2] if len(v) > 2 else 115.0)
    cramp = np.clip(((COOL[1] + 20) - np.arange(H, dtype=np.float32)[:, None]) / 40, 0, 1) * np.ones((1, W), np.float32)   # 1 above to_row
    if os.environ.get("COOL_SKIP"):
        x0, y0, x1, y1 = map(int, os.environ["COOL_SKIP"].split(",")); box = np.zeros((H, W), np.float32); box[y0:y1, x0:x1] = 1; cramp = cramp * (1 - cv2.GaussianBlur(box, (0, 0), 8))
head = None
if os.environ.get("KEEP"):
    hb = float(os.environ["HEAD_BELOW"]); yy = np.mgrid[0:H, 0:W][0]; head = (cv2.resize(cv2.imread(os.environ["KEEP"], cv2.IMREAD_GRAYSCALE), (W, H)) > 127) | (yy >= hb); sker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
elif os.environ.get("HEAD"):
    hcx, hcy, hrx, hry = map(float, os.environ["HEAD"].split(",")); hb = float(os.environ.get("HEAD_BELOW", hcy + hry))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); head = (((xx - hcx) / hrx) ** 2 + ((yy - hcy) / hry) ** 2 <= 1) | ((yy >= hcy) & (np.abs(xx - hcx) <= hrx)) | (yy >= hb); sker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
dec = ["ffmpeg", "-v", "error", "-i", inp, "-vf", f"fps={FPS},scale={W}:{H}:flags=lanczos", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
WHITE_LIFT = "format=yuv420p,lutyuv=y='if(gt(val,212),min(212+(val-212)*1.9,252),val)'"   # luma above 212 is stretched past 235: decodes as pure white
def enc(path, vf=None): return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", *(["-vf", vf] if vf else []),
    "-c:v", "libx264", "-preset", "medium", "-crf", "11", "-g", "1", "-bf", "0", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv", path], stdin=subprocess.PIPE)
p = subprocess.Popen(dec, stdout=subprocess.PIPE); qf = enc(pre + "-fg.mp4"); qm = enc(pre + "-mk.mp4", WHITE_LIFT)
prev = None; n = 0; t0 = time.time()
ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * OUT + 1, 2 * OUT + 1)) if OUT else None
white = np.full((H, W, 3), 255, np.float32)
while True:
    buf = p.stdout.read(W * H * 3)
    if len(buf) < W * H * 3: break
    f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    if n % EVERY == 0 or prev is None:
        x = cv2.resize(f, (mw, mh), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 127.5 - 1
        a = sess.run(None, {"input": x.transpose(2, 0, 1)[None]})[0][0, 0]
        a = cv2.resize(np.clip((a - 0.2) / 0.7, 0, 1), (W, H), interpolation=cv2.INTER_LINEAR)
        if seg is not None:
            r = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f[sy0:sy1, :, ::-1]))); cm = []
            for m in r.confidence_masks:   # 0 background, 1 hair, 2 body skin, 3 face skin, 4 clothes, 5 others (accessories)
                full = np.zeros((H, W), np.float32); full[sy0:sy1] = cv2.resize(np.squeeze(m.numpy_view()), (W, sy1 - sy0), interpolation=cv2.INTER_LINEAR); cm.append(full)
            gate = cv2.GaussianBlur(cv2.dilate(((cm[1] + cm[2] + cm[3] + cm[4] + cm[5]) > 0.5).astype(np.uint8), pker).astype(np.float32), (0, 0), 3)   # class 5: a clip-on mic, glasses, headphones
            if nramp is not None:
                gh = cv2.GaussianBlur(cv2.dilate(((cm[1] + cm[2] + cm[3]) > 0.5).astype(np.uint8), pker).astype(np.float32), (0, 0), 3); gate = gh * (1 - nramp) + gate * nramp
            gate[:sy0] = 1; gate[sy1:] = 1                                   # outside the segmenter's rows: the matte alone decides
            a = np.maximum(a * gate, cv2.GaussianBlur(np.clip((cm[4] + cm[2] + cm[5] - 0.35) / 0.4, 0, 1), (0, 0), 2) * bramp)
            if head is not None:
                skin = cv2.dilate(((cm[2] + cm[3]) > 0.5).astype(np.uint8), sker) > 0
                a = a * cv2.GaussianBlur((head | skin).astype(np.float32), (0, 0), 2.5)
        elif head is not None: a = a * cv2.GaussianBlur(head.astype(np.float32), (0, 0), 2.5)
        if COOL:   # dark AND cooler than the hair: a black chair back, a monitor, a dark sofa. A property of the pixels, so it does not flicker
            fb = f.astype(np.float32); lum = 0.114 * fb[..., 0] + 0.587 * fb[..., 1] + 0.299 * fb[..., 2]; dk = (lum < COOL[2]).astype(np.float32)
            warm = cv2.GaussianBlur((fb[..., 2] - fb[..., 0]) * dk, (0, 0), 8) / (cv2.GaussianBlur(dk, (0, 0), 8) + 1e-3)
            cd = dk * np.clip((COOL[0] + 1.5 - warm) / 3.0, 0, 1)                                  # 1 = dark and cool
            obj = cv2.dilate((cv2.GaussianBlur(cd, (0, 0), 7) > 0.22).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)   # the whole object, mesh gaps included
            a = a * (1 - cv2.GaussianBlur(obj, (0, 0), 2) * cramp)
        prev = a if prev is None else 0.65 * a + 0.35 * prev
        al = prev
        # keep only the largest connected blob (drops wall art / islands the matte sometimes grabs)
        nb, lab, st, _ = cv2.connectedComponentsWithStats((al > 0.5).astype(np.uint8), 8)
        if nb > 2:
            keep = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); blob = cv2.dilate((lab == keep).astype(np.uint8), np.ones((9, 9), np.uint8)); al = al * blob
        al = cv2.erode(al, np.ones((3, 3), np.uint8)); al = cv2.GaussianBlur(al, (0, 0), 1.3)
        if OUT:   # smooth sticker edge: blur the alpha first so hair wisps do not make the outline boil
            body = (cv2.GaussianBlur(al, (0, 0), 5) > 0.4).astype(np.uint8)
            ad = cv2.GaussianBlur(cv2.dilate(body, ker).astype(np.float32), (0, 0), 1.6); ad = np.maximum(ad, al)
        else: ad = al
        dn = np.roll(ad, 16, axis=0); dn[:16] = 0                             # shadow falls slightly below the sticker (no wrap-around)
        sh = cv2.GaussianBlur(dn, (0, 0), 26)
        mk = (1 - ad) * (1 - SH * sh)
        if SOLID < 1: al = np.where(ad > 0.5, np.power(al, SOLID), al)
        a3 = al[:, :, None]; o3 = (ad - al).clip(0, 1)[:, :, None]
    ff = f.astype(np.float32)
    # edge decontamination: semi-transparent pixels carry the old wall colour -> pull them toward the outline white
    edge = (1 - np.abs(al * 2 - 1))[:, :, None] * (0.5 if OUT else 0.0)
    ff = ff * (1 - edge) + white * edge
    fg = ff * a3 + white * o3
    qf.stdin.write(fg.clip(0, 255).astype(np.uint8).tobytes()); qm.stdin.write(cv2.merge([(mk * 255).clip(0, 255).astype(np.uint8)] * 3).tobytes()); n += 1
for q in (qf, qm): q.stdin.close(); q.wait()
print(f"{n} frames {n / (time.time() - t0):.1f} fps", file=sys.stderr)
