# voiceclean.py <take> <out.wav> — clean a talking-head take's voice ONCE, over the whole take, before cutting.
# The noise profile is taken from the room tone BETWEEN utterances (utt/<take>.json from utter.py), so the
# denoiser removes this room's noise and nothing else. A fixed noise floor guess (e.g. afftdn nf=-32 on a take
# whose real floor is -50 dB) eats quiet parts of the voice and sounds watery.
# Chain: mid of L/R -> spectral gate (noisereduce, stationary) -> high-pass, gentle EQ, one soft compressor ->
# static gain to TARGET LUFS -> limiter. No dynamic loudness normalisation (it pumps).
# Env: PROP (noise reduction 0..1, default 0.88), TARGET (LUFS, default -15), LIMIT_DB (default -3).
import sys, os, json, subprocess, re
import numpy as np, noisereduce as nr, scipy.io.wavfile as wavfile
src, out = sys.argv[1:3]; SR = 48000
name = os.path.basename(src).rsplit(".", 1)[0]; U = json.load(open(f"utt/{name}.json"))["utts"]
PROP = float(os.environ.get("PROP", 0.88)); TARGET = float(os.environ.get("TARGET", -15)); LIMIT = 10 ** (float(os.environ.get("LIMIT_DB", -3)) / 20)
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
x = np.frombuffer(raw, np.float32).reshape(-1, 2).mean(axis=1)
gaps = []; prev = 0.0
for u in U + [dict(s=len(x) / SR, e=len(x) / SR)]:
    if u["s"] - prev > 0.7: gaps.append((prev + 0.2, u["s"] - 0.2))
    prev = u["e"]
src_note = f"{len(gaps)} gaps sampled"
if sum(b - a for a, b in gaps) < 1.0:
    # A take read in one breath has no long gaps. Never fall back to the head of the file (it is usually speech, and a
    # speech "noise profile" makes the gate eat the voice): take the quietest 20 ms frames of the whole take instead.
    F = SR // 50; nfr = len(x) // F; fr = x[:nfr * F].reshape(nfr, F); lv = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)
    q = np.where(lv <= np.percentile(lv, 4) + 1.5)[0]; gaps = []
    for i in q:
        if gaps and abs(gaps[-1][1] - i * F / SR) < 1e-6: gaps[-1] = (gaps[-1][0], (i + 1) * F / SR)
        else: gaps.append((i * F / SR, (i + 1) * F / SR))
    src_note = f"no long gaps: quietest frames, {sum(b - a for a, b in gaps):.1f} s"
noise = np.concatenate([x[int(a * SR):int(b * SR)] for a, b in gaps])
db = lambda v: 20 * np.log10(np.sqrt(np.mean(v ** 2)) + 1e-12)
y = nr.reduce_noise(y=x, sr=SR, y_noise=noise, stationary=True, prop_decrease=PROP, n_fft=2048, win_length=2048, hop_length=512,
                    freq_mask_smooth_hz=400, time_mask_smooth_ms=80, n_std_thresh_stationary=1.2).astype(np.float32)
noise2 = np.concatenate([y[int(a * SR):int(b * SR)] for a, b in gaps])
tmp = out + ".dn.wav"; wavfile.write(tmp, SR, y)
chain = "highpass=f=75,equalizer=f=250:t=q:w=1.0:g=-2,equalizer=f=3500:t=q:w=1.2:g=2.5,highshelf=f=9000:g=1.5,acompressor=threshold=-26dB:ratio=3:attack=12:release=160:knee=6:makeup=1"
m = subprocess.run(["ffmpeg", "-nostats", "-i", tmp, "-af", chain + ",ebur128", "-f", "null", "-"], capture_output=True, text=True).stderr
I = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", m)[-1]); gain = TARGET - I
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-af", f"{chain},aresample=192000,volume={gain:.2f}dB,alimiter=limit={LIMIT:.3f}:level=false:attack=2:release=80,aresample={SR}", "-ac", "1", "-c:a", "pcm_s24le", out], check=True)
os.remove(tmp)
print(f"room tone {db(noise):.1f} dB -> {db(noise2):.1f} dB ({src_note}), loudness {I:.1f} LUFS, gain {gain:+.1f} dB -> {out}")
