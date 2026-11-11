# Background blur for talking-head video: MODNet alpha (512px), temporal EMA, blurred plate composited under the sharp person.
import sys, subprocess, numpy as np, cv2, onnxruntime as ort, time
inp, out = sys.argv[1], sys.argv[2]
ss, dur = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (None, None)
W, H, FPS = 1920, 1080, 30
so = ort.SessionOptions(); so.intra_op_num_threads = 4
import os; sess = ort.InferenceSession(os.environ.get("MODNET", os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "modnet.onnx")), so)
dec = ["ffmpeg", "-v", "error"] + (["-ss", str(ss), "-t", str(dur)] if ss is not None else []) + ["-i", inp, "-vf", f"fps={FPS}", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
enc = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-preset", "medium", "-crf", "12", "-g", "1", "-bf", "0", "-pix_fmt", "yuv420p", out]
p = subprocess.Popen(dec, stdout=subprocess.PIPE); q = subprocess.Popen(enc, stdin=subprocess.PIPE)
prev = None; n = 0; t0 = time.time()
mw, mh = int(__import__("os").environ.get("MW", 640)), int(__import__("os").environ.get("MH", 352))
big = None
while True:
    buf = p.stdout.read(W * H * 3)
    if len(buf) < W * H * 3: break
    f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    if n % 2 == 0 or prev is None:
        x = cv2.resize(f, (mw, mh), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 127.5 - 1
        a = sess.run(None, {"input": x.transpose(2, 0, 1)[None]})[0][0, 0]
        prev = a if prev is None else 0.6 * a + 0.4 * prev  # temporal smoothing
        al = cv2.resize(np.clip((prev - 0.08) / 0.84, 0, 1), (W, H), interpolation=cv2.INTER_LINEAR)
        al8 = cv2.merge([(al * 255).astype(np.uint8)] * 3)
    bg = cv2.GaussianBlur(cv2.resize(f, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 6)
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)
    o = cv2.add(cv2.multiply(f, al8, scale=1 / 255), cv2.multiply(bg, cv2.bitwise_not(al8), scale=1 / 255))
    q.stdin.write(o.tobytes()); n += 1
q.stdin.close(); q.wait()
print(f"{n} frames {n/(time.time()-t0):.1f} fps", file=sys.stderr)
