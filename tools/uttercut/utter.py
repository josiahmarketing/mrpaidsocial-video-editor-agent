# utter.py <take.mp4>... — split a raw take into utterances (speech runs separated by pauses >= GAP) and transcribe each
# one on its own (medium, word timestamps). Isolated short phrases make false starts, restarts and spoken "cut" markers
# visible; whole-take transcription tends to merge a false start into its retake. -> utt/<take>.json
import sys, os, json, subprocess
import numpy as np
from faster_whisper import WhisperModel
GAP = float(os.environ.get("GAP", 0.30))
m = WhisperModel(os.environ.get("MODEL", "medium"), device="cpu", compute_type="int8", cpu_threads=int(os.environ.get("THREADS", 4)))
os.makedirs("utt", exist_ok=True)
for src in sys.argv[1:]:
    name = os.path.basename(src).rsplit(".", 1)[0]; out = f"utt/{name}.json"
    if os.path.exists(out): print("SKIP", name); continue
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-af", "highpass=f=80,afftdn=nf=-30", "-f", "f32le", "-"], capture_output=True).stdout
    x = np.frombuffer(raw, np.float32)
    rms = 20 * np.log10(np.sqrt(np.convolve(x ** 2, np.ones(320) / 320, "same")[::160]) + 1e-9)
    p90 = float(np.percentile(rms, 90)); hi, lo = p90 - 28, p90 - 42
    on = rms >= hi; runs = []; st = None
    for i, v in enumerate(list(on) + [False]):
        if v and st is None: st = i
        if not v and st is not None: runs.append([st, i]); st = None
    runs = [r for r in runs if r[1] - r[0] >= 6]                      # >= 60 ms
    merged = []
    for a, b in runs:
        if merged and (a - merged[-1][1]) / 100 < GAP: merged[-1][1] = b
        else: merged.append([a, b])
    U = []
    for a, b in merged:
        # soft edges: walk outward while energy stays above the low threshold (catches soft onsets and word tails)
        s = a
        while s > 0 and rms[s - 1] >= lo and a - s < 25: s -= 1
        e = b
        while e < len(rms) and rms[e] >= lo and e - b < 35: e += 1
        seg = x[max(0, s * 160 - 1600): min(len(x), e * 160 + 1600)]   # transcribe with 0.1 s context either side
        segs, _ = m.transcribe(seg, language="en", word_timestamps=True, vad_filter=False, condition_on_previous_text=False,
                               initial_prompt="Um, uh. Cut. Okay, cut. Sorry.")
        words = [dict(w=w.word.strip(), s=round(max(0, s * 160 - 1600) / 16000 + w.start, 3), e=round(max(0, s * 160 - 1600) / 16000 + w.end, 3)) for q in segs for w in q.words]
        U.append(dict(s=round(s / 100, 3), e=round(e / 100, 3), text=" ".join(w["w"] for w in words), words=words))
    json.dump(dict(src=src, p90=p90, utts=U), open(out, "w"), indent=1)
    print("UTT", name, len(U), flush=True)
