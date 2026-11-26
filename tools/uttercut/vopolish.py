# vopolish.py <voice.wav> <words.json> <outdir> — polish a cut voice-over: real pauses between sentences, a quiet room.
#
# For a voice-over read fast and cut tight (pauses of 0.1-0.2 s, room tone and breath audible between words), it:
#   1. finds the quietest 5 ms point at every sentence / comma boundary (words.json gives the boundaries; whisper
#      starts run ~0.09 s early, so the search reaches past the next word's start), and inserts true silence there
#      to reach the target pause, with short fades. A boundary whose quietest point is louder than SKIP_DB is
#      speech running through (a run-on), and is left exactly as read: silence there would clip a word.
#   2. shifts every word time by the silence inserted before it -> <outdir>/words.json (captions stay on the words).
#   3. reduces room noise with a profile taken from this read's own quiet frames, EQ (-2.5 dB at 300 Hz, +2 dB at
#      3.5 kHz), de-ess, light compression, then a soft downward expander between words (up to 18 dB down, fast
#      attack, slow release so decaying word tails are never chopped), static gain to TARGET LUFS, limiter.
#      Order matters: compression BEFORE the gate, or the compressor lifts the room tone back up.
# Prints the pauses added and the room tone relative to speech before -> after. Check the result by transcribing
# it unprompted and diffing against words.json: nothing should be missing.
# Env: SENT (0.24 s), PUNCH_S (0.40 s), COMMA (0.10 s), PUNCH (comma-separated boundary times in the INPUT that get
#      PUNCH_S: the lines that land), SKIP_DB (-35), TARGET (-15 LUFS), PROP (0.85 noise reduction).
import sys, os, json, subprocess
import numpy as np, scipy.io.wavfile as wf, noisereduce as nr
src, wj, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
SENT = float(os.environ.get("SENT", 0.24)); PS = float(os.environ.get("PUNCH_S", 0.40)); COMMA = float(os.environ.get("COMMA", 0.10))
PUNCH = [float(x) for x in os.environ.get("PUNCH", "").split(",") if x.strip()]
SKIP = float(os.environ.get("SKIP_DB", -35)); TARGET = float(os.environ.get("TARGET", -15)); PROP = float(os.environ.get("PROP", 0.85))
SR = 48000; tmp = os.path.join(out, "_in.wav")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", str(SR), tmp], check=True)
_, a = wf.read(tmp); a = a.astype(np.float32) / 32768
def env(x, h):
    return 20 * np.log10(np.array([np.sqrt((x[i*h:(i+1)*h] ** 2).mean()) for i in range(len(x) // h)]) + 1e-9)
d = json.load(open(wj)); W = d.get("words", d) if isinstance(d, dict) else d

# 1. pauses
e5 = env(a, SR // 200); ins = []
for k, w in enumerate(W[:-1]):
    t = w["text"]
    if t[-1:] not in ".?!,": continue
    lo = max(0, int((w["end"] - 0.02) / 0.005)); hi = int((W[k+1]["start"] + 0.16) / 0.005)
    seg = e5[lo:hi]
    if not len(seg): continue
    i = int(np.argmin(seg)); m = seg[i]; L = R = i
    while L > 0 and seg[L-1] < m + 6: L -= 1
    while R < len(seg) - 1 and seg[R+1] < m + 6: R += 1
    cut = (lo + i) * 0.005
    if m > SKIP: print(f"  keep {cut:6.2f}  after {t!r}: speech runs through ({m:.0f} dB)"); continue
    tgt = COMMA if t.endswith(",") else (PS if any(abs(cut - p) < 0.03 for p in PUNCH) else SENT)
    add = tgt - (R - L + 1) * 0.005
    if add > 0.02: ins.append((cut, add, 0.012 if m < -45 else 0.025, t))
parts = []; prev = 0; shift = []; tot = 0.0
for c, dd, f, t in ins:
    i = int(c * SR); nf = int(f * SR); seg = a[prev:i].copy(); seg[-nf:] *= np.linspace(1, 0, nf) ** 2
    parts += [seg, np.zeros(int(dd * SR), np.float32)]; a[i:i+nf] *= np.linspace(0, 1, nf) ** 2
    prev = i; tot += dd; shift.append((c, tot))
parts.append(a[prev:]); y = np.concatenate(parts)
def mv(t):
    s = 0.0
    for c, cum in shift:
        if t >= c: s = cum
    return round(t + s, 3)
for w in W: w["start"], w["end"] = mv(w["start"]), mv(w["end"])
json.dump(d, open(os.path.join(out, "words.json"), "w"))
print(f"{len(ins)} pauses added, +{tot:.2f} s -> {len(y) / SR:.2f} s")

# 3. clean
H = SR // 100; e = env(y, H); live = e > -80; sp = np.percentile(e[live], 90)
q = np.where(live & (e < sp - 26))[0]
noise = np.concatenate([y[i*H:(i+1)*H] for i in q]) if len(q) else y[:SR // 2]
z = nr.reduce_noise(y=y, sr=SR, y_noise=noise, stationary=True, prop_decrease=PROP, n_fft=2048)
p1, p2 = os.path.join(out, "_nr.wav"), os.path.join(out, "_comp.wav")
wf.write(p1, SR, (np.clip(z, -1, 1) * 32767).astype(np.int16))
af = ("highpass=f=80,equalizer=f=300:t=q:w=1.2:g=-2.5,equalizer=f=3500:t=q:w=1.0:g=2,deesser=i=0.4:f=0.5,"
      "acompressor=threshold=0.1:ratio=2:attack=10:release=150:makeup=1")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", p1, "-af", af, p2], check=True)
_, c = wf.read(p2); c = c.astype(np.float32) / 32768
ec = env(c, H); thr = np.percentile(ec[ec > -80], 90) - 24
g = np.clip((ec - (thr - 12)) / 12, 0, 1); gdb = np.convolve(-18 * (1 - g), np.ones(5) / 5, "same")
gs = np.repeat(10 ** (gdb / 20), H); gs = np.pad(gs, (0, len(c) - len(gs)), "edge")
for i in range(1, len(gs)): gs[i] = max(gs[i], gs[i-1] * 0.9995)
p3 = os.path.join(out, "_gated.wav"); wf.write(p3, SR, (np.clip(c * gs, -1, 1) * 32767).astype(np.int16))
lines = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", p3, "-af", "ebur128", "-f", "null", "-"], capture_output=True, text=True).stderr.splitlines()
I = float([l for l in lines if " I:" in l][-1].split()[1]); fin = os.path.join(out, "voice.wav")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", p3, "-af", f"volume={TARGET - I:.2f}dB,alimiter=limit=0.708:level=false:attack=3:release=60", "-ar", str(SR), fin], check=True)
_, o = wf.read(fin); o = o.astype(np.float32) / 32768; eo = env(o, H); n = min(len(e), len(eo)); Q = (e[:n] > -80) & (e[:n] < sp - 24)
print(f"room tone under speech: {sp - np.median(e[:n][Q]):.1f} dB -> {np.percentile(eo[eo > -80], 90) - np.median(eo[:n][Q]):.1f} dB;  {fin}")
for f in (tmp, p1, p2, p3): os.remove(f)
