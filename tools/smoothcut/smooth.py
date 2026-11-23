# smooth.py <cut video> <out.mp4> — make a jump-cut talking head read as one continuous take.
# 1. Finds every cut in the file (frame-difference spikes and sudden jumps of the face).
# 2. Face-lock: each segment is reframed so the eyes sit at one place and size (EYE_X/EYE_Y, SCALE), and a slow drift
#    inside each segment makes the eyes of the last frame before a cut land exactly on the eyes of the first frame
#    after it. Head movement inside a segment is kept; only the jump at the cut is removed.
# 3. Morph: the MORPH frames on each side of a cut are replaced by optical-flow in-betweens of the frames around them,
#    so mouth, hair and shoulders glide instead of popping. Audio is not touched (same frame count).
# Output is OWxOH (1080x1920). Give it a source larger than the output so the reframing has room; edges it cannot
# cover are mirrored and then blurred (SOFT px sigma, 0 = off), because a sharp mirror of a sleeve or a chair back reads
# as a kaleidoscope pattern at the frame edge. Env: MODEL (face_landmarker.task), SCALE (output px per source px at the median face size,
# default 0.853), EYE_X / EYE_Y preferred eye position (540 / 720; each segment gets as near as its framing allows),
# HPAD / VPAD px of mirrored edge allowed at the sides / bottom to hold that position (0), TPAD the same at the top
# (0; a speaker placed smaller or lower than the source frames, e.g. for a cut-out, needs all three), MORPH (3),
# CUTS="f1,f2" extra cut frames, QA=dir (a strip of frames around every cut + meta.json).
# OUT="1920x1080" sets the output size (default 1080x1920); AW is the width the face is tracked at (540; use 960 for landscape).
# SMIN=<scale> is the smallest zoom a segment may get (default none): with takes where the speaker sits at two distances,
# SMIN=1.0 stops the closer one from being shrunk until the source no longer covers the frame.
# GRADE="<ffmpeg filters>" applies a colour grade in the same encode (one generation, no extra pass over the picture).
import sys, os, json, subprocess, collections, numpy as np, cv2, mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
src, out = sys.argv[1:3]; FPS = 30; OW, OH = map(int, os.environ.get("OUT", "1080x1920").split("x"))   # OUT=1920x1080 for landscape
W, H = map(int, subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", src], capture_output=True, text=True).stdout.strip().split(","))
S0 = float(os.environ.get("SCALE", 0.853)); EY = float(os.environ.get("EYE_Y", 720)); MORPH = int(os.environ.get("MORPH", 3)); QA = os.environ.get("QA")
fl = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=os.environ["MODEL"]), num_faces=1))
# ---------- pass 1: eyes per frame + frame differences
aw = int(os.environ.get("AW", 540)); ah = aw * H // W   # analysis width: raise it (AW=960) when the face is small in a wide frame
p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-vf", f"scale={aw}:{ah}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
mid = []; dist = []; D = [0.0]; prev = None
while True:
    b = p.stdout.read(aw * ah * 3)
    if len(b) < aw * ah * 3: break
    f = np.frombuffer(b, np.uint8).reshape(ah, aw, 3); r = fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f)))
    if r.face_landmarks:
        L = r.face_landmarks[0]; re = np.array([(L[33].x + L[133].x) / 2 * W, (L[33].y + L[133].y) / 2 * H]); le = np.array([(L[362].x + L[263].x) / 2 * W, (L[362].y + L[263].y) / 2 * H])
        mid.append((re + le) / 2); dist.append(float(np.linalg.norm(le - re)))
    else: mid.append(np.array([np.nan, np.nan])); dist.append(np.nan)
    g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (180, max(2, 180 * ah // aw))).astype(np.float32)
    if prev is not None: D.append(float(np.abs(g - prev).mean()))
    prev = g
N = len(mid); mid = np.array(mid); dist = np.array(dist); D = np.array(D); idx = np.arange(N); okf = ~np.isnan(dist)
for k in range(2): mid[:, k] = np.interp(idx, idx[okf], mid[okf, k])
dist = np.interp(idx, idx[okf], dist[okf]); print(f"{N} frames, face found in {okf.sum()}", file=sys.stderr)
# ---------- cuts: a one-frame spike in picture change, or a face jump far above its neighbours
J = np.r_[0, np.linalg.norm(np.diff(mid, axis=0), axis=1) / dist[1:]]; cuts = set(int(c) for c in os.environ.get("CUTS", "").split(",") if c)
for i in range(3, N - 3):
    nbD = max(D[i - 2], D[i - 1], D[i + 1], D[i + 2]); nbJ = max(J[i - 2], J[i - 1], J[i + 1], J[i + 2])
    if (D[i] > 3.0 and D[i] > 2.2 * nbD) or (J[i] > 0.06 and J[i] > 3.0 * nbJ and D[i] > 1.25 * nbD) or (D[i] > 2.2 and D[i] > 1.7 * nbD and J[i] > 2.0 * nbJ and J[i] > 0.03): cuts.add(i)
if os.environ.get("DEBUG"):
    for i in [int(v) for v in os.environ["DEBUG"].split(",")]: print("dbg", i, [(j, round(float(D[j]), 1), round(float(J[j]), 3)) for j in range(i - 3, i + 4)], file=sys.stderr)
cuts = sorted(cuts); cuts = [c for j, c in enumerate(cuts) if j == 0 or c - cuts[j - 1] > 2 * MORPH + 2]
edges = [0] + cuts + [N]; K = len(edges) - 1; print(f"{len(cuts)} cuts at frames {cuts}", file=sys.stderr)
# light smoothing of the eye track inside each segment (landmark jitter must not shake the frame)
for a, b in zip(edges[:-1], edges[1:]):
    if b - a >= 5:
        for arr in (mid[:, 0], mid[:, 1], dist):
            v = arr[a:b].copy(); q = np.pad(v, 2, mode="edge"); arr[a:b] = (q[:-4] + 2 * q[1:-3] + 3 * q[2:-2] + 2 * q[3:-1] + q[4:]) / 9
Dref = float(np.median(dist)); med = np.array([np.median(mid[a:b], axis=0) for a, b in zip(edges[:-1], edges[1:])]); s = np.array([S0 * Dref / np.median(dist[a:b]) for a, b in zip(edges[:-1], edges[1:])])
s = np.maximum(s, float(os.environ.get("SMIN", 0)))   # SMIN=1.0: a segment where the speaker leans in keeps its larger face instead of shrinking past the frame edge
EX = float(os.environ.get("EYE_X", OW / 2)); MG = float(os.environ.get("MARGIN", 45)); Tk = np.zeros((K, 2))
HP = float(os.environ.get("HPAD", 0)); VP = float(os.environ.get("VPAD", 0)); TP = float(os.environ.get("TPAD", 0))     # px of mirrored edge allowed at the sides / at the bottom / at the top
for k in range(K):
    lox, hix = OW - s[k] * (W - med[k, 0]) + MG - HP, s[k] * med[k, 0] - MG + HP; loy, hiy = OH - s[k] * (H - med[k, 1]) + MG - VP, s[k] * med[k, 1] - MG + TP
    Tk[k] = [np.clip(EX, lox, hix) if lox <= hix else (lox + hix) / 2, np.clip(EY, loy, hiy) if loy <= hiy else (loy + hiy) / 2]
# ---------- boundary corrections: scale first, then position
rs = np.ones(K); re_ = np.ones(K); ds = np.zeros((K, 2)); de = np.zeros((K, 2))
for k in range(K - 1):
    c = edges[k + 1]; qe = s[k] * dist[max(c - 2, edges[k]):c].mean(); qs = s[k + 1] * dist[c:min(c + 2, edges[k + 2])].mean(); m = np.sqrt(qe * qs); re_[k] = m / qe; rs[k + 1] = m / qs
for k in range(K - 1):
    c = edges[k + 1]; pe = Tk[k] + s[k] * re_[k] * (mid[max(c - 2, edges[k]):c].mean(axis=0) - med[k]); ps = Tk[k + 1] + s[k + 1] * rs[k + 1] * (mid[c:min(c + 2, edges[k + 2])].mean(axis=0) - med[k + 1])
    m = (pe + ps) / 2; de[k] = m - pe; ds[k + 1] = m - ps
Ms = np.zeros((N, 2, 3)); over = 0.0
for k in range(K):
    a, b = edges[k], edges[k + 1]
    for i in range(a, b):
        u = (i - a) / max(1, b - 1 - a); u = u * u * (3 - 2 * u); sg = s[k] * np.exp((1 - u) * np.log(rs[k]) + u * np.log(re_[k])); off = (1 - u) * ds[k] + u * de[k]
        tx, ty = Tk[k] + off - sg * med[k]; Ms[i] = [[sg, 0, tx], [0, sg, ty]]
        over = max(over, tx, OW - (sg * W + tx), OH - (sg * H + ty), ty)          # > 0: the source does not cover the output there
print(f"eye target x {Tk[:, 0].min():.0f}..{Tk[:, 0].max():.0f}, y {Tk[:, 1].min():.0f}..{Tk[:, 1].max():.0f}, scale {s.min():.3f}..{s.max():.3f}, cut corrections up to {max(np.abs(ds).max(), np.abs(de).max()):.0f} px, uncovered edge up to {max(over, 0):.0f} px (mirrored)", file=sys.stderr)
# ---------- pass 2: reframe, morph across cuts, encode
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); gx, gy = np.meshgrid(np.arange(OW, dtype=np.float32), np.arange(OH, dtype=np.float32))
def flow(a, b):
    f = dis.calc(cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (OW // 2, OH // 2)), cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (OW // 2, OH // 2)), None)
    f = cv2.GaussianBlur(f, (0, 0), 4) * 2; f = cv2.resize(f, (OW, OH), interpolation=cv2.INTER_LINEAR); mag = np.linalg.norm(f, axis=2, keepdims=True); return f * np.minimum(1, 70 / (mag + 1e-6))
def morph(A, B, n):
    fab, fba = flow(A, B), flow(B, A); res = []
    A1 = cv2.remap(A, gx - fab[..., 0], gy - fab[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)       # A pushed all the way to B
    r = cv2.GaussianBlur(np.abs(A1.astype(np.float32) - B).mean(axis=2), (0, 0), 12); hard = np.clip((r - 14) / 26, 0, 1)[:, :, None]   # 1 = flow failed here
    for j in range(n):
        t = (j + 1) / (n + 1); t = t * t * (3 - 2 * t); tf = float(np.clip((t - 0.5) * 3.2 + 0.5, 0, 1)); w = (1 - hard) * t + hard * tf
        At = cv2.remap(A, gx - t * fab[..., 0], gy - t * fab[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); Bt = cv2.remap(B, gx - (1 - t) * fba[..., 0], gy - (1 - t) * fba[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        res.append((At * (1 - w) + Bt * w).clip(0, 255).astype(np.uint8))
    return res
dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-vf", "scale=in_color_matrix=bt709:in_range=tv,format=bgr24", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-", "-vf", (os.environ["GRADE"] + "," if os.environ.get("GRADE") else "") + "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
    "-c:v", "libx264", "-preset", "medium", "-crf", "13", "-g", "30", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv", out], stdin=subprocess.PIPE)
buf = collections.OrderedDict(); cset = {c + MORPH: c for c in cuts}; strips = {}; hold = 2 * MORPH + 2
def emit(i, f):
    enc.stdin.write(f.tobytes())
    if QA:
        for c in cuts:
            if c - 5 <= i <= c + 4: strips.setdefault(c, []).append(cv2.resize(f, (216, 384) if OH > OW else (320, 180)))
SOFT = float(os.environ.get("SOFT", 16)); ONES = np.full((H, W), 255, np.uint8)
def soften(f, M):   # blur the mirrored band, feathered over 20 px from the real picture
    if SOFT <= 0: return f
    m = cv2.warpAffine(ONES, M, (OW, OH), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if m.all(): return f
    a = np.clip(cv2.distanceTransform((m == 0).astype(np.uint8), cv2.DIST_L2, 3) / 20, 0, 1)[..., None]
    return (f * (1 - a) + cv2.GaussianBlur(f, (0, 0), SOFT) * a).astype(np.uint8)
for i in range(N):
    b = dec.stdout.read(W * H * 3)
    if len(b) < W * H * 3: break
    buf[i] = soften(cv2.warpAffine(np.frombuffer(b, np.uint8).reshape(H, W, 3), Ms[i], (OW, OH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101), Ms[i])
    if i in cset and MORPH > 0:
        c = cset[i]; A, B = buf[c - MORPH - 1], buf[i]
        for j, m in enumerate(morph(A, B, 2 * MORPH)): buf[c - MORPH + j] = m
    while buf and next(iter(buf)) <= i - hold: k0, f0 = buf.popitem(last=False); emit(k0, f0)
while buf: k0, f0 = buf.popitem(last=False); emit(k0, f0)
enc.stdin.close(); enc.wait(); print(f"{N} frames -> {out}", file=sys.stderr)
if QA:
    os.makedirs(QA, exist_ok=True); rows = []
    for c in cuts:
        r = np.hstack(strips[c]); cv2.putText(r, f"cut @ frame {c} ({c / FPS:.2f}s)", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2); rows.append(r)
    for j in range(0, len(rows), 4): cv2.imwrite(os.path.join(QA, f"cuts-{j // 4:02d}.jpg"), np.vstack(rows[j:j + 4]))
    json.dump({"cuts": cuts, "eye_x": [float(Tk[:, 0].min()), float(Tk[:, 0].max())], "eye_y": [float(Tk[:, 1].min()), float(Tk[:, 1].max())], "scale": [float(s.min()), float(s.max())], "frames": N}, open(os.path.join(QA, "meta.json"), "w"))
