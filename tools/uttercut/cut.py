# cut.py <part> — build <part>/mezz.mov from chosen UTTERANCES (see utter.py) butted together with no pauses.
# plan.json (or $PLAN): { part: { "take": "14-27-50", "vf": optional crop, "keep": [ 0, 3, {"u": 17, "from": "and most of these"},
#                                                                   {"u": 11, "to": "ask me"}, ... ] } }
#   int                -> the whole utterance (its soft onset and its tail until the voice has died away)
#   {"u", "from"}      -> start at the LAST occurrence of that word sequence inside the utterance (drops a restart)
#   {"u", "to"}        -> end after the FIRST occurrence of that word sequence
#   {"u", "start"/"end"} -> absolute source seconds, for edges the word timestamps get wrong (verify with headchk.py)
#   {"u", "pre": 1.6}  -> also keep that many SOURCE seconds before the utterance's (tightened) start, untouched: the
#                         speaker listening before they answer, for a second voice laid over it in the edit. Continuous
#                         footage, so there is no join between the listening and the answer.
#   {"range": [a, b]}  -> a verbatim source range with no utterance (a listening or reaction shot from elsewhere in the take)
# Each piece is tightened to the voice (first/last frame above p90-EDGE_DB) with LEAD before and TAIL after, so sentences
# finish but room tone between them is gone; inner pauses longer than MAXPAUSE are shrunk the same way (0 = keep them).
import sys, os, json, re, subprocess, math
import numpy as np
k = sys.argv[1]; P = json.load(open(os.environ.get("PLAN", "plan.json")))[k]
U = json.load(open(f"utt/{P['take']}.json")); src = U["src"]; utts = U["utts"]
AUDIO = os.environ.get("AUDIO")   # optional pre-cleaned voice track on the take's timeline (see voiceclean.py)
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", AUDIO or src, "-vn", "-ac", "1", "-ar", "16000"] + ([] if AUDIO else ["-af", "highpass=f=80,afftdn=nf=-30"]) + ["-f", "f32le", "-"], capture_output=True).stdout
x = np.frombuffer(raw, np.float32); rms = 20 * np.log10(np.sqrt(np.convolve(x ** 2, np.ones(320) / 320, "same")[::160]) + 1e-9)
lo = U["p90"] - 42
_m = np.zeros(len(rms), bool)
for _u in utts: _m[int(_u["s"] * 100):int(_u["e"] * 100)] = True
# room tone between utterances. A take read in one breath has almost no gaps, and the few there are hold breaths and word
# tails: measuring "noise" there puts the threshold above quiet sentence-final words and the cutter drops them as pauses.
# With under 2 s of gaps, use the quietest 1 % of the whole take instead (in such a take even the 10th percentile is speech).
noise = float(np.percentile(rms[~_m], 90)) if (~_m).sum() > 200 else float(np.percentile(rms, 1))
speech = float(np.percentile(rms[_m], 90)) if _m.sum() > 50 else U["p90"] + 3   # voice level inside utterances
thr = max(speech - float(os.environ.get("EDGE_DB", 35)), noise + 8); LEAD = float(os.environ.get("LEAD", 0.07)); TAIL = float(os.environ.get("TAIL", 0.13)); MAXPAUSE = float(os.environ.get("MAXPAUSE", 0.32))
n = lambda t: re.sub(r"[^a-z0-9$%']", "", t.lower())
def find(words, phrase, last):
    toks = [n(t) for t in phrase.split()]; hits = [i for i in range(len(words) - len(toks) + 1) if all(n(words[i + j]["w"]) == toks[j] for j in range(len(toks)))]
    if not hits: raise SystemExit(f"{k}: phrase '{phrase}' not in: {' '.join(w['w'] for w in words)}")
    return (hits[-1] if last else hits[0]), len(toks)
def quiet_before(t, reach=0.15):   # lowest-energy 10 ms frame in [t-reach, t]
    a, b = int((t - reach) * 100), int(t * 100); return (a + int(np.argmin(rms[a:b + 1]))) / 100 if b > a else t
def tail(t, cap=0.35):
    i = int(t * 100); j = i
    while j < len(rms) and rms[j] >= lo and (j - i) / 100 < cap: j += 1
    return j / 100 + 0.06
segs = []
for item in P["keep"]:
    if isinstance(item, dict) and "range" in item: segs.append([round(math.floor(item["range"][0] * 30) / 30, 4), round(math.ceil(item["range"][1] * 30) / 30, 4)]); continue
    if isinstance(item, int): u = utts[item]; s, e = u["s"] - 0.04, u["e"] + 0.06
    else:
        u = utts[item["u"]]; W = u["words"]; s, e = u["s"] - 0.04, u["e"] + 0.06
        if "from" in item:   # word starts from whisper run late: search back to the previous word's end
            i, _ = find(W, item["from"], True); prev = W[i - 1]["e"] if i else W[i]["s"] - 0.15
            s = quiet_before(W[i]["s"], min(0.25, max(0.06, W[i]["s"] - prev)))
        if "start" in item: s = item["start"]
        if "end" in item: e = item["end"]
        if "to" in item:
            i, L = find(W, item["to"], False); e = tail(W[i + L - 1]["e"])
            if i + L < len(W): e = min(e, W[i + L]["s"] - 0.03)   # never run into the next word
    # tighten to the voice: room tone above the splitter's low threshold can pad an utterance by ~0.3 s each side.
    # Keep LEAD before the first voiced frame and TAIL after the last, and shrink inner pauses longer than MAXPAUSE.
    explicit_s = isinstance(item, dict) and ("from" in item or "start" in item); explicit_e = isinstance(item, dict) and ("to" in item or "end" in item)
    a0, b0 = max(0, int(s * 100)), min(len(rms), int(e * 100)); v = np.where(rms[a0:b0] >= thr)[0]
    if len(v):
        if not explicit_s: s = max(s, (a0 + v[0]) / 100 - LEAD)
        if not explicit_e: e = min(e, (a0 + v[-1]) / 100 + TAIL)
    parts = [[s, e]]
    if MAXPAUSE > 0 and len(v):
        on = rms[int(s * 100):int(e * 100)] >= thr; st = None; base = int(s * 100); out = []; cur = s
        for i, x in enumerate(list(on) + [True]):
            if not x and st is None: st = i
            if x and st is not None:
                if (i - st) / 100 > MAXPAUSE and st > 0 and i < len(on): out.append([cur, (base + st) / 100 + TAIL]); cur = (base + i) / 100 - LEAD
                st = None
        out.append([cur, e]); parts = [p for p in out if p[1] - p[0] > 0.08]
    if isinstance(item, dict) and item.get("pre"): parts[0][0] -= float(item["pre"])
    for s, e in parts: segs.append([round(math.floor(s * 30) / 30, 4), round(math.ceil(e * 30) / 30, 4)])   # floor starts, ceil ends: never shave a frame of speech
for i in range(1, len(segs)):
    if segs[i - 1][0] < segs[i][0] < segs[i - 1][1]: segs[i][0] = segs[i - 1][1]
SPEED = float(os.environ.get("SPEED", 1))   # e.g. 1.1: pictures are sampled 1.1x apart from the source, the voice is time-stretched (pitch kept)
t = 0; edl = []
for i, (s, e) in enumerate(segs): od = round((e - s) * 30 / SPEED) / 30; edl.append(dict(src_start=s, src_end=e, out_start=round(t, 4), out_end=round(t + od, 4), win=i)); t += od
os.makedirs(k, exist_ok=True)
json.dump(dict(fps=30, src=src, speed=SPEED, segments=edl), open(f"{k}/edl.json", "w"), indent=1)
print(k, len(segs), "pieces", round(t, 2), "s", flush=True)
if os.environ.get("DRY"): sys.exit()
crop = P.get("vf", "scale=1920:1080:flags=lanczos")
os.makedirs(f"{k}/segs", exist_ok=True); lst = []
for i, (s, e) in enumerate(segs):   # PICTURE ONLY per piece: sound encoded alongside it loses its last milliseconds (see pieces.py)
    d = e - s; nf = round(d * 30 / SPEED); f = f"{k}/segs/{i:03d}.mov"; lst.append(f"file 'segs/{i:03d}.mov'")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s:.4f}", "-t", f"{d + 0.2:.4f}", "-i", src, "-an", "-frames:v", str(nf), "-r", "30",
                    "-vf", f"fps={30 / SPEED:.6f},{crop},setsar=1,format=yuv420p,setpts=N/(30*TB)",
                    "-c:v", "libx264", "-preset", "fast", "-crf", os.environ.get("MEZZ_CRF", "10"), "-g", "1", "-bf", "0", f], check=True)   # MEZZ_CRF=15 halves the mezzanine when disk is short
open(f"{k}/list.txt", "w").write("\n".join(lst) + "\n")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import pieces
pcm = pieces.build(dict(speed=SPEED, segments=edl), AUDIO or src); pieces.write_wav(f"{k}/cut.wav", pcm)   # every piece exactly frames * 1600 samples
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", f"{k}/list.txt", "-i", f"{k}/cut.wav", "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                "-af", os.environ.get("AFINAL", "aresample=48000" if AUDIO else "highpass=f=80,afftdn=nf=-32,acompressor=threshold=-20dB:ratio=2.5:attack=8:release=120,loudnorm=I=-16:TP=-1.5:LRA=7,aresample=48000"),
                "-c:a", "pcm_s16le", f"{k}/mezz.mov"], check=True)
dv, da = [float(subprocess.run(["ffprobe", "-v", "error", "-select_streams", x, "-show_entries", "stream=duration", "-of", "csv=p=0", f"{k}/mezz.mov"], capture_output=True, text=True).stdout.split()[0]) for x in ("v:0", "a:0")]
if abs(dv - da) > 0.01: raise SystemExit(f"{k}: picture {dv:.3f} s but sound {da:.3f} s - the cut would drift out of sync")
os.remove(f"{k}/cut.wav")
import shutil; shutil.rmtree(f"{k}/segs"); os.remove(f"{k}/list.txt")
print("mezz ok", k, flush=True)
