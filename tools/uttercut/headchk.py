# headchk.py <src> <t1> <t2> ... — transcribe 1.6 s of <src> from each start time (small model) to hear the first words
import sys, subprocess, numpy as np
from faster_whisper import WhisperModel
m = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=2)
for a in sys.argv[2:]:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", a, "-t", "1.8", "-i", sys.argv[1], "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True).stdout
    s, _ = m.transcribe(np.frombuffer(raw, np.float32), language="en", vad_filter=False)
    print(a, "|", " ".join(t.text.strip() for t in s), flush=True)
