# track.py <video> <out.json> — pass 1 of the print blur: follow a printed graphic on clothing through a talking-head cut.
# A clean reference frame (REF_T seconds, print fully visible) gives the print's rectangle and its SIFT features; every
# frame is matched back to that reference (affine transform, RANSAC), so hands crossing the print do not pull the
# track. Frames with too few matches are bridged between good neighbours inside the same shot, following a small bright
# marker on the same garment when one is set (MARK = "h_lo,h_hi" hue range of e.g. a mic's LED; optional).
# Env: ROI "x,y,w,h" search region, REF_T, SEED "x,y" a point inside the print in the reference frame, CUTS "t1,t2,..."
# shot changes in seconds, MININ minimum RANSAC inliers (default 8).
import sys, os, json, subprocess, numpy as np, cv2
src, out = sys.argv[1:3]; W, H, FPS = 1080, 1920, 30
rx, ry, rw, rh = map(int, os.environ.get("ROI", "110,1060,860,860").split(",")); REF_T = float(os.environ.get("REF_T", 1.0))
sx, sy = map(int, os.environ.get("SEED", "570,1650").split(",")); CUTS = [float(c) for c in os.environ.get("CUTS", "").split(",") if c]; MININ = int(os.environ.get("MININ", 8))
MARK = [int(v) for v in os.environ.get("MARK", "").split(",") if v]
def frames():
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    while True:
        b = p.stdout.read(W * H * 3)
        if len(b) < W * H * 3: break
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3)
raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(REF_T), "-i", src, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
ref = np.frombuffer(raw, np.uint8).reshape(H, W, 3); roi = ref[ry:ry + rh, rx:rx + rw]
# the print = the bright connected region around the seed point (dark garment around it)
br = (roi.max(axis=2) > 85).astype(np.uint8); br = cv2.morphologyEx(br, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(br, 8); k = lab[sy - ry, sx - rx]
if k == 0: raise SystemExit("seed is not on the print in the reference frame")
x, y, w, h = st[k, :4]
# a hand or arm touching the print in the reference frame widens the box: take the left/right edges from the top third only
top = (lab[y:y + max(40, h // 3), x:x + w] == k).mean(axis=0) > 0.5; cols = np.where(top)[0]
if len(cols) > 40: x, w = x + int(cols[0]), int(cols[-1] - cols[0] + 1)
rect = [int(x + rx), int(y + ry), int(w), int(h)]
sift = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.03); g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY); pm = np.zeros((rh, rw), np.uint8); pm[y + 4:y + h - 2, x + 4:x + w - 4] = 255
k0, d0 = sift.detectAndCompute(g, pm); bf = cv2.BFMatcher(cv2.NORM_L2); p0 = np.float32([kp.pt for kp in k0])
print(f"reference {REF_T}s: print rect {rect}, {len(k0)} features", file=sys.stderr)
T = []; marks = []
for i, f in enumerate(frames()):
    r = f[ry:ry + rh, rx:rx + rw]; kp, d = sift.detectAndCompute(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY), None); t = None
    if d is not None and len(kp) >= 8:
        m = [a for a, b in bf.knnMatch(d0, d, k=2) if a.distance < 0.78 * b.distance]
        if len(m) >= MININ:
            # full affine: cloth shears and foreshortens as the speaker moves, which a rotation+scale model reads as tilt
            A, inl = cv2.estimateAffine2D(p0[[a.queryIdx for a in m]], np.float32([kp[a.trainIdx].pt for a in m]), method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=3000)
            if A is not None and inl.sum() >= MININ:
                sxx, syy = float(np.hypot(A[0, 0], A[1, 0])), float(np.hypot(A[0, 1], A[1, 1]))
                if 0.75 < sxx < 1.3 and 0.75 < syy < 1.3 and abs(A[0, 1]) < 0.4 and abs(A[1, 0]) < 0.4: t = [float(v) for v in A.ravel()] + [int(inl.sum())]
    T.append(t)
    if MARK:
        hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV); mk = ((hsv[..., 0] >= MARK[0]) & (hsv[..., 0] <= MARK[1]) & (hsv[..., 1] > 150) & (hsv[..., 2] > 170)).astype(np.uint8)
        nn, ll, ss, cc = cv2.connectedComponentsWithStats(mk, 8); best = None
        for j in range(1, nn):
            if 5 <= ss[j, 4] <= 600 and (best is None or ss[j, 4] > ss[best, 4]): best = j
        marks.append([float(cc[best][0]), float(cc[best][1])] if best else None)
    if i % 300 == 0: print(i, t, file=sys.stderr)
# drop a track that jumps away from both neighbours' tracks (a wrong match on a mostly covered print)
ctr = lambda t: np.array([t[0] * (rw / 2) + t[1] * (rh / 2) + t[2], t[3] * (rw / 2) + t[4] * (rh / 2) + t[5]])
for i in range(1, len(T) - 1):
    if T[i] and T[i - 1] and T[i + 1] and np.linalg.norm(ctr(T[i]) - (ctr(T[i - 1]) + ctr(T[i + 1])) / 2) > 30 and np.linalg.norm(ctr(T[i - 1]) - ctr(T[i + 1])) < 30: T[i] = None
good = sum(t is not None for t in T); print(f"{len(T)} frames, {good} tracked directly ({100 * good / len(T):.0f} %)", file=sys.stderr)
json.dump({"roi": [rx, ry, rw, rh], "rect": rect, "ref_t": REF_T, "cuts": CUTS, "track": T, "marks": marks}, open(out, "w"))
