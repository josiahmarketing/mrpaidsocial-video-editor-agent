# cutout.py in.mp4 out_prefix — "sticker cutout" of a talking head for HTML compositions, as TWO plain videos that
# composite with CSS blend modes (no alpha channel needed, so any renderer that paints <video> frames can use them):
#   <prefix>-mk.mp4  multiply layer: white outside the sticker, black inside, with a soft drop shadow around it
#   <prefix>-fg.mp4  add layer: the speaker, premultiplied on black, plus a white sticker outline
# Stack them as SIBLINGS of the background graphics (same stacking context):
#   <video class="mk" style="mix-blend-mode:multiply">   then   <video class="fg" style="mix-blend-mode:plus-lighter">
# result = background x (1 - sticker) + speaker. Transform both videos with the same tween; never wrap them in a
# transformed/faded container (that isolates the group and the blend sees no backdrop).
# Env: MODNET (model path), OUTLINE px (default 12; 0 = none), SHADOW 0..1 (default 0.32), EVERY (matte every Nth frame).
import sys, os, subprocess, time, numpy as np, cv2, onnxruntime as ort
inp, pre = sys.argv[1:3]
W, H, FPS = 1080, 1920, 30
OUT = int(os.environ.get("OUTLINE", 12)); SH = float(os.environ.get("SHADOW", 0.32)); EVERY = int(os.environ.get("EVERY", 1))
so = ort.SessionOptions(); so.intra_op_num_threads = int(os.environ.get("THREADS", 4))
sess = ort.InferenceSession(os.environ.get("MODNET", os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "modnet.onnx")), so)
dec = ["ffmpeg", "-v", "error", "-i", inp, "-vf", f"fps={FPS},scale={W}:{H}:flags=lanczos", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
def enc(path): return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
    "-c:v", "libx264", "-preset", "medium", "-crf", "11", "-g", "1", "-bf", "0", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv", path], stdin=subprocess.PIPE)
p = subprocess.Popen(dec, stdout=subprocess.PIPE); qf = enc(pre + "-fg.mp4"); qm = enc(pre + "-mk.mp4")
mw, mh = 384, 672; prev = None; n = 0; t0 = time.time()
ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * OUT + 1, 2 * OUT + 1)) if OUT else None
white = np.full((H, W, 3), 255, np.float32)
while True:
    buf = p.stdout.read(W * H * 3)
    if len(buf) < W * H * 3: break
    f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    if n % EVERY == 0 or prev is None:
        x = cv2.resize(f, (mw, mh), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 127.5 - 1
        a = sess.run(None, {"input": x.transpose(2, 0, 1)[None]})[0][0, 0]
        prev = a if prev is None else 0.65 * a + 0.35 * prev
        al = cv2.resize(np.clip((prev - 0.2) / 0.7, 0, 1), (W, H), interpolation=cv2.INTER_LINEAR)
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
        a3 = al[:, :, None]; o3 = (ad - al).clip(0, 1)[:, :, None]
    ff = f.astype(np.float32)
    # edge decontamination: semi-transparent pixels carry the old wall colour -> pull them toward the outline white
    edge = (1 - np.abs(al * 2 - 1))[:, :, None] * (0.5 if OUT else 0.0)
    ff = ff * (1 - edge) + white * edge
    fg = ff * a3 + white * o3
    qf.stdin.write(fg.clip(0, 255).astype(np.uint8).tobytes()); qm.stdin.write(cv2.merge([(mk * 255).clip(0, 255).astype(np.uint8)] * 3).tobytes()); n += 1
for q in (qf, qm): q.stdin.close(); q.wait()
print(f"{n} frames {n / (time.time() - t0):.1f} fps", file=sys.stderr)
