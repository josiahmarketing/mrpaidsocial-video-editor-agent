# mkcanvas.py <part> — the iPad side of the cut. For every piece of <part>/edl.json take the screen recording at the same
# moments (plus a per-piece shift from shifts.json: he often draws the underline just AFTER the phrase, in a pause the cut
# removes, so the stroke is slid under the words), make each frame the clean steady canvas (canvas.py) and dissolve the
# joins. The recording is decoded ONCE, start to end, and only the needed frames come out (seeking a long-GOP screen
# recording per piece is very slow). -> <part>/public/canvas.mp4 (plate size, 30 fps, same frame count as the cut)
#   OFFSET        seconds the screen recording runs ahead of the camera take (screen time = camera time + OFFSET). Two
#                 recordings started "together" by hand are easily a second apart: measure it on a sync mark (README).
#   regions.json  {"piece": [[x0, y0, x1, y1], ...]} plate rectangles shown as the clean plate during that piece: a stroke
#                 left over from an earlier section that the speaker's own zoom now cuts in half, or a whole piece with
#                 no drawing in it ([[0, 0, W, H]] gives the plate itself, sharper than the recording).
import sys, os, json, subprocess, numpy as np, cv2
from canvas import Canvas, FW, FH
k = sys.argv[1]; E = json.load(open(f"{k}/edl.json")); SP = E.get("speed", 1); R = 30 / SP; N = 5
OFFSET = float(os.environ.get("OFFSET", 0))
SH = {int(a): b for a, b in json.load(open("shifts.json")).items()} if os.path.exists("shifts.json") else {}
RG = {int(a): b for a, b in json.load(open("regions.json")).items()} if os.path.exists("regions.json") else {}
need = []                                                    # (first source frame at R fps, frames) per piece
for i, g in enumerate(E["segments"]): need.append((round((g["src_start"] + OFFSET + SH.get(i, 0)) * R), round((g["out_end"] - g["out_start"]) * 30)))
for i in range(1, len(need)):                                # pieces must not run backwards for a single pass
    if need[i][0] < need[i - 1][0] + need[i - 1][1]: need[i] = (need[i - 1][0] + need[i - 1][1], need[i][1])
sel = "+".join(f"between(n\\,{a}\\,{a + n - 1})" for a, n in need)
dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", os.environ.get("SRC", "dl/raw.mp4"), "-vf", f"fps={R:.6f},select='{sel}'", "-vsync", "0", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
c = Canvas()
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{c.PW}x{c.PH}", "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "14",
                        "-g", "30", "-bf", "0", "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", f"{k}/public/canvas.mp4"], stdin=subprocess.PIPE)
prev = None; total = 0
def regmask(rects):                                          # 1 where the plate replaces the recording, soft-edged
    m = np.zeros((c.PH, c.PW), np.float32)
    for x0, y0, x1, y1 in rects: m[max(0, y0):y1, max(0, x0):x1] = 1
    return cv2.GaussianBlur(m, (0, 0), 4)[..., None]
for i, (a, nf) in enumerate(need):
    c.last = None; hold = prev.astype(np.float32) if prev is not None else None; o = None; got = 0
    rm = regmask(RG[i]) if i in RG else None
    while got < nf:
        b = dec.stdout.read(FW * FH * 3)
        if len(b) < FW * FH * 3: break
        fr = np.frombuffer(b, np.uint8).reshape(FH, FW, 3); o = c.frame(fr, markup=(a + got) / R >= float(os.environ.get("MARKUP_FROM", 0)))   # MARKUP_FROM: second at which the app's bars and toolbar appear
        if rm is not None: o = (o * (1 - rm) + c.plate * rm).astype(np.uint8)
        w = o if hold is None or got >= N else (hold * (1 - (got + 1) / (N + 1)) + o.astype(np.float32) * ((got + 1) / (N + 1))).astype(np.uint8)
        enc.stdin.write(w.tobytes()); got += 1
    while got < nf and o is not None: enc.stdin.write(o.tobytes()); got += 1
    prev = o; total += got; print(i, "src", round(a / R, 2), "frames", got, "fit", tuple(round(x, 3) for x in (c.last or (0, 0, 0))), flush=True)
enc.stdin.close(); enc.wait(); print("canvas frames", total)
