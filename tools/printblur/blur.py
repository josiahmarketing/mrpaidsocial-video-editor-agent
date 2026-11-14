# blur.py <video> <track.json> <out.mp4> — pass 2 of the print blur: blur ONLY the printed graphic, keep hands sharp.
# Per frame: the tracked print area (dilated, so the edge error of the track does not matter) is replaced by a heavy
# blur of the frame, except where a hand is in front of it. "Hand" = the skin class of a person segmenter, with a veto:
# a print that shows a person is itself labelled skin. So inside that part of the print (found once, on the clean
# reference frame) skin pixels whose local structure matches the tracked reference print (normalised cross-correlation,
# best over a small shift, solid regions only) are taken out, and what is left only counts as hand if its blob reaches
# skin outside the print. A printed person no hand touches is therefore always blurred; whole hands stay sharp.
# Env: SEG (segmenter .tflite), DIL print dilation px (40), NCC match threshold (0.55), QA=dir writes before/after sheets,
#      ONLY="t0-t1,..." process just these seconds and skip the encode (for tuning).
import sys, os, json, subprocess, numpy as np, cv2, mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
src, tj, out = sys.argv[1:4]; W, H, FPS = 1080, 1920, 30
D = json.load(open(tj)); rx, ry, rw, rh = D["roi"]; X, Y, RW, RH = D["rect"]; T = D["track"]; CUTS = [round(c * FPS) for c in D.get("cuts", [])]
DIL = int(os.environ.get("DIL", 40)); NCC = float(os.environ.get("NCC", 0.6)); QA = os.environ.get("QA")
ONLY = [tuple(float(v) * FPS for v in r.split("-")) for r in os.environ.get("ONLY", "").split(",") if r]
seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(base_options=mpp.BaseOptions(model_asset_path=os.environ["SEG"]), output_category_mask=True))
def skin_of(roi_bgr):
    cm = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(roi_bgr[:, :, ::-1]))).category_mask.numpy_view()
    return cv2.resize(cm, (rw, rh), interpolation=cv2.INTER_NEAREST)
DEC = ["-vf", "scale=in_color_matrix=bt709:in_range=tv,format=bgr24"]
raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(D["ref_t"]), "-i", src, "-frames:v", "1"] + DEC + ["-f", "rawvideo", "-"], capture_output=True).stdout
ref = np.frombuffer(raw, np.uint8).reshape(H, W, 3)[ry:ry + rh, rx:rx + rw].copy()
rect = np.zeros((rh, rw), np.uint8); rect[max(0, Y - ry):Y - ry + RH, X - rx:X - rx + RW] = 255
cm = skin_of(ref); Z = ((cm != 0) & (cm != 4) & (rect > 0)).astype(np.uint8) * 255; Z = cv2.dilate(Z, np.ones((49, 49), np.uint8))   # the person inside the print
hw, hh = rw // 2, rh // 2; refg = cv2.cvtColor(cv2.resize(ref, (hw, hh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
refc = cv2.blur(cv2.resize(ref, (hw, hh), interpolation=cv2.INTER_AREA), (5, 5)); CD = float(os.environ.get("CD", 13))
KW = (15, 15); box = lambda a: cv2.blur(a, KW); KO = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
# fill the frames the tracker could not match: interpolate inside each shot, hold at shot edges
n = len(T); A = np.full((n, 6), np.nan); ok = np.zeros(n, bool)
for i, t in enumerate(T):
    if t: A[i] = t[:6]; ok[i] = True
edges = [0] + [c for c in CUTS if 0 < c < n] + [n]
for a, b in zip(edges[:-1], edges[1:]):
    idx = np.arange(a, b); g = idx[ok[a:b]]
    if len(g) == 0: g = np.where(ok)[0]; A[a:b] = A[g[np.argmin(np.abs(g - a))]]; continue
    for k in range(6): A[a:b, k] = np.interp(idx, g, A[g, k])
    for k in range(6):                                   # light smoothing inside the shot: the track jitters by a pixel or two
        v = A[a:b, k].copy(); p = np.pad(v, 2, mode="edge"); A[a:b, k] = (p[:-4] + 2 * p[1:-3] + 3 * p[2:-2] + 2 * p[3:-1] + p[4:]) / 9
dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src] + DEC + ["-f", "rawvideo", "-"], stdout=subprocess.PIPE)
enc = None if ONLY else subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
    "-c:v", "libx264", "-preset", "medium", "-crf", "13", "-g", "30", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv", out], stdin=subprocess.PIPE)
k5 = np.ones((5, 5), np.uint8); kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DIL + 1, 2 * DIL + 1)); kd2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DIL + 41, 2 * DIL + 41))
qa = []; i = -1
while True:
    b = dec.stdout.read(W * H * 3)
    if len(b) < W * H * 3: break
    i += 1
    if ONLY and not any(a <= i <= z for a, z in ONLY): continue
    f = np.frombuffer(b, np.uint8).reshape(H, W, 3).copy(); roi = f[ry:ry + rh, rx:rx + rw]; M = A[i].reshape(2, 3); sure = ok[i]
    pm = cv2.dilate(cv2.warpAffine(rect, M, (rw, rh), flags=cv2.INTER_NEAREST), kd if sure else kd2)
    cm = skin_of(roi); skin = ((cm == 2) | (cm == 3)).astype(np.uint8)
    zw = cv2.warpAffine(Z, M, (rw, rh), flags=cv2.INTER_NEAREST)
    if not sure: zw = cv2.dilate(zw, np.ones((41, 41), np.uint8))
    # (1) structure: where does the frame show the tracked reference print? local NCC at half resolution, best over a small shift
    Mh = M.copy(); Mh[:, 2] /= 2; g = cv2.warpAffine(refg, Mh, (hw, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    fg = cv2.cvtColor(cv2.resize(roi, (hw, hh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    mf = box(fg); vf = np.maximum(box(fg * fg) - mf * mf, 0); R = 3 if sure else 14; st = 1 if sure else 2; flat = vf < 30   # smooth skin cannot "match" the print
    gp = cv2.copyMakeBorder(g, R, R, R, R, cv2.BORDER_REPLICATE); mgp = box(gp); vgp = np.maximum(box(gp * gp) - mgp * mgp, 0); best = np.full((hh, hw), -1, np.float32)
    for dy in range(0, 2 * R + 1, st):
        for dx in range(0, 2 * R + 1, st):
            gs = gp[dy:dy + hh, dx:dx + hw]; mg = mgp[dy:dy + hh, dx:dx + hw]; vg = vgp[dy:dy + hh, dx:dx + hw]
            ncc = (box(fg * gs) - mf * mg) / np.sqrt(vf * vg + 1e-3); ncc[(vg < 30) | flat] = -1; np.maximum(best, ncc, out=best)     # flat reference or flat frame: no evidence
    # the veto: print-like structure inside the print's person area, as solid regions (chance matches on a hand are specks)
    veto = ((best >= (NCC if sure else NCC + 0.14)) & (cv2.resize(zw, (hw, hh), interpolation=cv2.INTER_NEAREST) > 0)).astype(np.uint8)
    # flat parts of the print (a plain wall, a sheet) carry no structure: there, "same colour as the reference" marks print
    c0 = cv2.warpAffine(refc, Mh, (hw, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    cd = np.abs(cv2.blur(cv2.resize(roi, (hw, hh), interpolation=cv2.INTER_AREA).astype(np.float32), (5, 5)) - c0).mean(axis=2)
    inr = cv2.warpAffine(rect, Mh, (hw, hh), flags=cv2.INTER_NEAREST) > 0; vg0 = vgp[R:R + hh, R:R + hw]
    veto |= (inr & (vg0 < 30) & (cd < CD) & sure).astype(np.uint8)
    veto = cv2.dilate(cv2.morphologyEx(veto, cv2.MORPH_OPEN, KO), np.ones((3, 3), np.uint8)); veto = cv2.resize(veto, (rw, rh), interpolation=cv2.INTER_NEAREST)
    trusted = skin & (pm == 0)                      # skin well outside the print: certainly the speaker
    cand = skin & (pm > 0) & (1 - veto)
    nn, lab = cv2.connectedComponents((trusted | cand).astype(np.uint8), connectivity=8); hand = np.zeros((rh, rw), np.uint8)
    if nn > 1:
        cnt = np.bincount(lab[trusted > 0], minlength=nn); keep = np.where(cnt >= 150)[0]; keep = keep[keep > 0]; hand = np.isin(lab, keep).astype(np.uint8)
    hand = cv2.erode(cv2.morphologyEx(hand, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)), k5)          # blur reaches a couple of px under the hand's edge
    m = ((pm > 0) & (hand == 0)).astype(np.float32); m = cv2.GaussianBlur(m, (0, 0), 3.0)[:, :, None]
    sm = cv2.resize(roi, (rw // 14, rh // 14), interpolation=cv2.INTER_AREA); sm = cv2.GaussianBlur(sm, (0, 0), 2.4); bl = cv2.resize(sm, (rw, rh), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    o = (roi.astype(np.float32) * (1 - m) + bl * m).clip(0, 255).astype(np.uint8)
    if QA and i % (6 if ONLY else 12) == 0:
        v = roi.copy(); c, _ = cv2.findContours((pm > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(v, c, -1, (0, 255, 0) if sure else (0, 165, 255), 3)
        c, _ = cv2.findContours(hand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(v, c, -1, (255, 0, 255), 2)
        t3 = np.hstack([cv2.resize(v, (300, 300)), cv2.resize(o, (300, 300))]); cv2.putText(t3, f"{i / FPS:.1f}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2); qa.append(t3)
    if enc: f[ry:ry + rh, rx:rx + rw] = o; enc.stdin.write(f.tobytes())
    if i % 300 == 0: print(i, file=sys.stderr)
if enc: enc.stdin.close(); enc.wait(); print(f"{i + 1} frames -> {out}", file=sys.stderr)
if QA:
    os.makedirs(QA, exist_ok=True)
    for s in range(0, len(qa), 24):
        g = qa[s:s + 24]; g += [np.zeros_like(g[0])] * ((-len(g)) % 4); cv2.imwrite(os.path.join(QA, f"blur-{s // 24:02d}.jpg"), np.vstack([np.hstack(g[r:r + 4]) for r in range(0, len(g), 4)]))
