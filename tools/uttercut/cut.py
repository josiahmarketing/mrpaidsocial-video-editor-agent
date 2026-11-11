# cut.py <part> — build <part>/mezz.mov from chosen UTTERANCES (see utter.py) butted together with no pauses.
# plan.json (or $PLAN): { part: { "take": "14-27-50", "vf": optional crop, "keep": [ 0, 3, {"u": 17, "from": "and most of these"},
#                                                                   {"u": 11, "to": "ask me"}, ... ] } }
#   int                -> the whole utterance (its soft onset and its tail until the voice has died away)
#   {"u", "from"}      -> start at the LAST occurrence of that word sequence inside the utterance (drops a restart)
#   {"u", "to"}        -> end after the FIRST occurrence of that word sequence
#   {"u", "start"/"end"} -> absolute source seconds, for edges the word timestamps get wrong (verify with headchk.py)
# Each piece keeps a 40 ms lead and its natural tail (energy above p90-42 dB, max +0.35 s, plus 60 ms), so sentences finish.
import sys, os, json, re, subprocess, math
import numpy as np
k = sys.argv[1]; P = json.load(open(os.environ.get("PLAN", "plan.json")))[k]
U = json.load(open(f"utt/{P['take']}.json")); src = U["src"]; utts = U["utts"]
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-af", "highpass=f=80,afftdn=nf=-30", "-f", "f32le", "-"], capture_output=True).stdout
x = np.frombuffer(raw, np.float32); rms = 20 * np.log10(np.sqrt(np.convolve(x ** 2, np.ones(320) / 320, "same")[::160]) + 1e-9)
lo = U["p90"] - 42
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
    segs.append([round(math.floor(s * 30) / 30, 4), round(math.ceil(e * 30) / 30, 4)])   # floor starts, ceil ends: never shave a frame of speech
for i in range(1, len(segs)):
    if segs[i - 1][0] < segs[i][0] < segs[i - 1][1]: segs[i][0] = segs[i - 1][1]
t = 0; edl = []
for i, (s, e) in enumerate(segs): edl.append(dict(src_start=s, src_end=e, out_start=round(t, 4), out_end=round(t + e - s, 4), win=i)); t += e - s
os.makedirs(k, exist_ok=True)
json.dump(dict(fps=30, src=src, segments=edl), open(f"{k}/edl.json", "w"), indent=1)
print(k, len(segs), "pieces", round(t, 2), "s", flush=True)
if os.environ.get("DRY"): sys.exit()
crop = P.get("vf", "scale=1920:1080:flags=lanczos")
os.makedirs(f"{k}/segs", exist_ok=True); lst = []
for i, (s, e) in enumerate(segs):
    d = e - s; nf = round(d * 30); f = f"{k}/segs/{i:03d}.mov"; lst.append(f"file 'segs/{i:03d}.mov'")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s:.4f}", "-i", src, "-frames:v", str(nf), "-t", f"{nf / 30:.4f}",
                    "-vf", f"fps=30,{crop},setsar=1,format=yuv420p",
                    "-af", f"aformat=sample_rates=48000:channel_layouts=stereo,afade=t=in:d=0.01,afade=t=out:st={max(0, nf / 30 - 0.012):.4f}:d=0.012",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "10", "-g", "1", "-bf", "0", "-c:a", "pcm_s16le", f], check=True)
open(f"{k}/list.txt", "w").write("\n".join(lst) + "\n")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", f"{k}/list.txt", "-c:v", "copy",
                "-af", "highpass=f=80,afftdn=nf=-32,acompressor=threshold=-20dB:ratio=2.5:attack=8:release=120,loudnorm=I=-16:TP=-1.5:LRA=7,aresample=48000",
                "-c:a", "pcm_s16le", f"{k}/mezz.mov"], check=True)
import shutil; shutil.rmtree(f"{k}/segs"); os.remove(f"{k}/list.txt")
print("mezz ok", k, flush=True)
