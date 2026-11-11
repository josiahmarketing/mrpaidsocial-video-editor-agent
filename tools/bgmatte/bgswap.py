# bgswap.py in.mov out.mp4 bg.png — vertical "green screen": MODNet matte (temporal EMA) composites the speaker over a
# heavily blurred, darkened, slowly drifting version of bg.png. 1080x1920 @30, all-intra, video only.
import sys, subprocess, time, numpy as np, cv2, onnxruntime as ort
inp, out, bgp = sys.argv[1:4]
W, H, FPS = 1080, 1920, 30
so = ort.SessionOptions(); so.intra_op_num_threads = 4
import os; sess = ort.InferenceSession(os.environ.get("MODNET", os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "modnet.onnx")), so)
bg0 = cv2.imread(bgp); bh, bw = bg0.shape[:2]
s = max(W * 1.25 / bw, H * 1.25 / bh); bg0 = cv2.resize(bg0, (int(bw * s), int(bh * s)), interpolation=cv2.INTER_AREA)
bg0 = cv2.GaussianBlur(bg0, (0, 0), 22); bg0 = (bg0.astype(np.float32) * 0.5 + np.array([40, 8, 8], np.float32) * 0.5 * 0.35).clip(0, 255).astype(np.uint8)
dec = ["ffmpeg", "-v", "error", "-i", inp, "-vf", f"fps={FPS},scale={W}:{H}:flags=lanczos", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
enc = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-preset", "medium", "-crf", "12", "-g", "1", "-bf", "0", "-pix_fmt", "yuv420p", out]
p = subprocess.Popen(dec, stdout=subprocess.PIPE); q = subprocess.Popen(enc, stdin=subprocess.PIPE)
mw, mh = 384, 672; prev = None; n = 0; t0 = time.time(); N = 31 * FPS
while True:
    buf = p.stdout.read(W * H * 3)
    if len(buf) < W * H * 3: break
    f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    if n % 2 == 0 or prev is None:
        x = cv2.resize(f, (mw, mh), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 127.5 - 1
        a = sess.run(None, {"input": x.transpose(2, 0, 1)[None]})[0][0, 0]
        prev = a if prev is None else 0.6 * a + 0.4 * prev
        al = cv2.resize(np.clip((prev - 0.2) / 0.7, 0, 1), (W, H), interpolation=cv2.INTER_LINEAR); al = cv2.erode(al, np.ones((3, 3), np.uint8)); al = cv2.GaussianBlur(al, (0, 0), 1.2)
        al8 = cv2.merge([(al * 255).astype(np.uint8)] * 3); alk = al
    k = min(1.0, n / N)  # slow drift: pan down and zoom slightly over the clip
    bh2, bw2 = bg0.shape[:2]; z = 1.0 + 0.06 * k; cw, ch = int(W / 1.0 / z * (bw2 / W) / (bw2 / W)), int(H / z)
    ox = int((bw2 - cw) / 2); oy = int((bh2 - ch) * (0.2 + 0.5 * k))
    bg = cv2.resize(bg0[oy:oy + ch, ox:ox + cw], (W, H), interpolation=cv2.INTER_LINEAR)
    # edge decontamination: semi-transparent pixels carry the old wall colour -> pull them toward the background tone
    edge = cv2.merge([((1 - np.abs(alk * 2 - 1)) * 255).astype(np.uint8)] * 3)
    fd = cv2.addWeighted(f, 0.45, bg, 0.55, 0); f2 = cv2.add(cv2.multiply(fd, edge, scale=1 / 255), cv2.multiply(f, cv2.bitwise_not(edge), scale=1 / 255))
    o = cv2.add(cv2.multiply(f2, al8, scale=1 / 255), cv2.multiply(bg, cv2.bitwise_not(al8), scale=1 / 255))
    q.stdin.write(o.tobytes()); n += 1
q.stdin.close(); q.wait()
print(f"{n} frames {n / (time.time() - t0):.1f} fps", file=sys.stderr)
