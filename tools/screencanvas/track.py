# track.py — for every 1/15 s of the screen recording: where is the drawing on screen (scale + offset against the clean plate),
# is the markup toolbar up and where, and how much new ink appeared. -> track.json
# The canvas only ever scales and slides (pinch zoom), so the fit is a similarity without rotation, found from ORB matches.
import subprocess, json, os, numpy as np, cv2, sys
SRC = os.environ.get("SRC", "dl/raw.mp4"); FW, FH = map(int, os.environ.get("FRAME", "2732x2048").split("x")); FPS = 15
plate = cv2.imread("plate.png"); PH, PW = plate.shape[:2]; K = 0.5
pg = cv2.resize(cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY), None, fx=K, fy=K, interpolation=cv2.INTER_AREA)
orb = cv2.ORB_create(4000, scaleFactor=1.2, nlevels=10); kp0, d0 = orb.detectAndCompute(pg, None); bf = cv2.BFMatcher(cv2.NORM_HAMMING)
p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", SRC, "-vf", f"fps={FPS},scale={int(FW*K)}:{int(FH*K)}", "-pix_fmt", "gray", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
W, H = int(FW * K), int(FH * K); out = []; n = 0; last = None; prev_ink = None
while True:
    b = p.stdout.read(W * H)
    if len(b) < W * H: break
    g = np.frombuffer(b, np.uint8).reshape(H, W); rec = dict(t=round(n / FPS, 4))
    kp, d = orb.detectAndCompute(g, None); fit = None
    if d is not None and len(kp) > 30:
        m = [a for a, c in bf.knnMatch(d0, d, k=2) if a.distance < 0.75 * c.distance] if len(kp) > 1 else []
        if len(m) >= 12:
            src = np.float32([kp0[a.queryIdx].pt for a in m]) / K; dst = np.float32([kp[a.trainIdx].pt for a in m]) / K
            M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4)
            if M is not None and inl.sum() >= 10:
                s = float(np.hypot(M[0, 0], M[1, 0])); rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
                fit = dict(s=round(s, 5), tx=round(float(M[0, 2]), 2), ty=round(float(M[1, 2]), 2), rot=round(rot, 3), inl=int(inl.sum()))
    rec["fit"] = fit
    # toolbar: a big dark-grey rounded bar (not black ink, not white paper)
    mid = ((g > 38) & (g < 105)).astype(np.uint8); mid = cv2.morphologyEx(mid, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    nlab, lab, st, _ = cv2.connectedComponentsWithStats(mid, 8); tb = None
    for i in range(1, nlab):
        x, y, w, h, a = st[i]
        if w > 0.35 * W and 0.03 * H < h < 0.2 * H and a > 0.5 * w * h: tb = [int(x / K), int(y / K), int((x + w) / K), int((y + h) / K)]
    rec["toolbar"] = tb
    ink = g < 110; rec["ink"] = int(ink.sum()); rec["new"] = int((ink & ~prev_ink).sum()) if prev_ink is not None else 0; prev_ink = ink
    out.append(rec); n += 1
json.dump(out, open("track.json", "w")); print(n, "samples", sum(1 for r in out if r["fit"]), "fitted", sum(1 for r in out if r["toolbar"]), "with toolbar")
