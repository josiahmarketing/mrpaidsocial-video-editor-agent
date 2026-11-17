# pieces.py — the audio of a cut, built sample-exactly from its EDL.
#
# Why this exists: encoding each piece with picture and sound in one ffmpeg call ("-frames:v N" plus "apad,atrim") ends the
# file when the picture is done, and the last few milliseconds of sound are still in the filter chain. Every piece came out
# 0-22 ms short at its tail, and butting the pieces together slid everything after each join earlier: about 5 ms per join,
# a third of a second over 70 joins. Lips and voice drifted apart through the video and no single join looked wrong.
#
# So the sound is never taken from the piece files. Each piece is decoded on its own, padded or trimmed to EXACTLY
# frames * 1600 samples (48 kHz, 30 fps), faded 10 ms in / 12 ms out, and the arrays are joined here.
import subprocess, wave, numpy as np

SR = 48000; SPF = SR // 30   # samples per output frame


def piece(src, s, d, nf, speed=1.0, pre=None):
    """Stereo int16 audio of one piece: source seconds [s, s+d) played at `speed`, exactly nf frames long."""
    af = (pre + "," if pre else "") + "aformat=sample_rates=48000:channel_layouts=stereo" + (f",atempo={speed}" if speed != 1 else "")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{s:.4f}", "-t", f"{d:.4f}", "-i", src, "-vn", "-af", af, "-f", "s16le", "-ac", "2", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.int16).reshape(-1, 2).astype(np.float32); n = nf * SPF
    a = a[:n] if len(a) >= n else np.vstack([a, np.zeros((n - len(a), 2), np.float32)])
    fi, fo = min(n, SR // 100), min(n, int(SR * 0.012))
    a[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)[:, None]; a[n - fo:] *= np.linspace(1, 0, fo, dtype=np.float32)[:, None]
    return np.clip(np.rint(a), -32768, 32767).astype(np.int16)


def build(edl, src, pre=None):
    """All pieces of an EDL (as written by cut.py) joined; piece i starts at sample round(out_start * 48000)."""
    sp = edl.get("speed", 1); out = []
    for g in edl["segments"]:
        nf = round((g["out_end"] - g["out_start"]) * 30); out.append(piece(src, g["src_start"], g["src_end"] - g["src_start"], nf, sp, pre))
    return np.vstack(out)


def write_wav(path, pcm):
    with wave.open(path, "wb") as w: w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())


def mono(path, extra=()):
    return np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", *extra, "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout, np.float32)


def offsets(edl, src, mixed, pre=None, reach=0.8):
    """Where each piece's voice sits in `mixed` (a path) against where its picture sits: [(index, out_start, ms, corr)].
    Negative ms = the voice is early. Works on a finished master too: music and effects only lower the correlation."""
    hay = mono(mixed); sp = edl.get("speed", 1); R = int(reach * SR); res = []
    for i, g in enumerate(edl["segments"]):
        nf = round((g["out_end"] - g["out_start"]) * 30); ref = piece(src, g["src_start"], g["src_end"] - g["src_start"], nf, sp, pre).astype(np.float32).mean(1) / 32768
        exp = int(round(g["out_start"] * SR)); h0 = min(SR // 20, len(ref) // 4); ch = ref[h0:h0 + int(0.6 * SR)]
        lo = max(0, exp - R); seg = hay[lo:exp + len(ref) + R]
        if len(ch) < SR // 10 or len(seg) <= len(ch) or not np.any(ch): res.append((i, g["out_start"], None, 0.0)); continue
        c = np.correlate(seg, ch, "valid"); j = int(np.argmax(np.abs(c)))
        res.append((i, g["out_start"], (lo + j - h0 - exp) / SR * 1000, float(abs(c[j]) / (np.linalg.norm(ch) * np.linalg.norm(seg[j:j + len(ch)]) + 1e-9))))
    return res
