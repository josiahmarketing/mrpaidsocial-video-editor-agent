# canvas.py — a tablet screen recording of someone drawing, turned into a clean, steady canvas: every frame is fitted to the clean plate of the drawing
# (scale + offset, so his own pinch zooms disappear), the app's bars and the markup toolbar are replaced by the plate, and
# his live strokes stay. Output is in plate coordinates, so one set of focus boxes works for the whole video.
#   as a module: Canvas().frame(bgr_frame, markup=True) -> canvas image (PH x PW)
#   as a script: canvas.py test t1 t2 ...   writes _qa/canvas-<t>.jpg for quick checks
import sys, os, subprocess, numpy as np, cv2
FW, FH = map(int, os.environ.get("FRAME", "2732x2048").split("x"))   # the screen recording's frame size
class Canvas:
    def __init__(self, plate="plate.png"):
        self.plate = cv2.imread(plate); self.PH, self.PW = self.plate.shape[:2]; self.K = 0.5
        # PATCH="x0,y0,x1,y1" paints one rectangle of the plate white (a viewer button caught in the plate, for example)
        if os.environ.get("PATCH"): x0, y0, x1, y1 = map(int, os.environ["PATCH"].split(",")); self.plate[y0:y1, x0:x1] = 255
        pg = cv2.resize(cv2.cvtColor(self.plate, cv2.COLOR_BGR2GRAY), None, fx=self.K, fy=self.K, interpolation=cv2.INTER_AREA)
        self.orb = cv2.ORB_create(4000, scaleFactor=1.2, nlevels=10); self.kp0, self.d0 = self.orb.detectAndCompute(pg, None); self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
    def fit(self, frame):
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=self.K, fy=self.K, interpolation=cv2.INTER_AREA)
        kp, d = self.orb.detectAndCompute(g, None)
        if d is not None and len(kp) > 30:
            m = [a for a, c in self.bf.knnMatch(self.d0, d, k=2) if a.distance < 0.75 * c.distance]
            if len(m) >= 12:
                src = np.float32([self.kp0[a.queryIdx].pt for a in m]) / self.K; dst = np.float32([kp[a.trainIdx].pt for a in m]) / self.K
                M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3)
                if M is not None and inl.sum() >= 10:
                    f = (float(np.hypot(M[0, 0], M[1, 0])), float(M[0, 2]), float(M[1, 2]))
                    # a still view should not shimmer: keep the previous fit unless this one really moved
                    if self.last and abs(f[0] - self.last[0]) < 0.0015 and abs(f[1] - self.last[1]) < 1.2 and abs(f[2] - self.last[2]) < 1.2: f = self.last
                    self.last = f; return f
        return self.last
    def toolbar(self, frame):
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        mid = ((g > 38) & (g < 105)).astype(np.uint8); mid = cv2.morphologyEx(mid, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(mid, 8); boxes = []
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if w > 0.3 * g.shape[1] and h > 0.02 * g.shape[0] and a > 0.4 * w * h: boxes.append((x * 4, y * 4, (x + w) * 4, (y + h) * 4))
        return boxes
    def frame(self, frame, markup=True):
        f = self.fit(frame)
        if f is None: return self.plate.copy()
        s, tx, ty = f; ok = np.full((FH, FW), 255, np.uint8)
        if markup:
            ok[:int(os.environ.get("CHROME", 196))] = 0                      # status bar, title, buttons (CHROME = their height in px)
            for x0, y0, x1, y1 in self.toolbar(frame): ok[max(0, y0 - 40):y1 + 40, max(0, x0 - 40):x1 + 40] = 0
        M = np.float32([[s, 0, tx], [0, s, ty]])                           # plate -> frame
        warped = cv2.warpAffine(frame, M, (self.PW, self.PH), flags=cv2.INTER_AREA if s > 1.05 else cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=(255, 255, 255)) if False else \
                 cv2.warpAffine(frame, M, (self.PW, self.PH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        mask = cv2.warpAffine(ok, M, (self.PW, self.PH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.GaussianBlur(cv2.erode(mask, np.ones((9, 9), np.uint8)), (0, 0), 3).astype(np.float32)[..., None] / 255
        return (warped * mask + self.plate * (1 - mask)).astype(np.uint8)
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "test":
    c = Canvas()
    for t in sys.argv[2:]:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", t, "-i", os.environ.get("SRC", "dl/raw.mp4"), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
        fr = np.frombuffer(raw, np.uint8).reshape(FH, FW, 3); c.last = None; o = c.frame(fr, markup=float(t) >= float(os.environ.get("MARKUP_FROM", 0)))
        cv2.imwrite(f"_qa/canvas-{t}.jpg", cv2.resize(o, (1168, 880), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88]); print(t, c.last)
