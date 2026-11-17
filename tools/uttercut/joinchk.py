# joinchk.py <part> — transcribe ~2.3 s across every join of the cut (<part>/mezz.mov + <part>/edl.json) with a second,
# smaller model and print what is heard there. Read the list top to bottom: a spoken "cut" that the utterance transcript
# missed, a clipped first word ("anything" for "nothing") or a doubled word ("So, so") shows up at the join it belongs to.
# Lines that contain "cut" are flagged with <<<<. Fix each one with {"u": N, "start": t} / {"u": N, "end": t} and re-cut.
import sys, json, subprocess, numpy as np
from faster_whisper import WhisperModel
k = sys.argv[1]; E = json.load(open(f"{k}/edl.json"))["segments"]; m = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=3)
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{k}/mezz.mov", "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True).stdout
x = np.frombuffer(raw, np.float32)
for i, s in enumerate(E):
    t = max(0, s["out_start"] - 0.45); a = x[int(t * 16000):int((t + 2.3) * 16000)]
    segs, _ = m.transcribe(a, language="en", initial_prompt="Okay, cut. Um, uh.", vad_filter=False)
    txt = " ".join(g.text.strip() for g in segs)
    flag = "  <<<<" if "cut" in txt.lower().split() or "cut." in txt.lower() or "cut," in txt.lower() else ""
    print(f"{i:2d} out {s['out_start']:6.2f} src {s['src_start']:7.2f} | {txt}{flag}", flush=True)
