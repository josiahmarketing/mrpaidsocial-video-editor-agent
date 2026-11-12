# remix.py <index.html> <out.wav> [--check render.mp4] [name=volume ...]
# Rebuild a HyperFrames composition's audio mix with ffmpeg from its <audio> tags (src, data-start, data-duration,
# data-volume), so an audio-only revision (a level change, a swapped sound) does not need a full video re-render:
# remix, master, then mux the new audio onto the already-rendered video with -c:v copy.
#   --check render.mp4   null-test the rebuilt mix against the renderer's own audio first. A residual 25 dB or more
#                        under the signal means the two mixes are the same and the remix can be trusted.
#   name=volume          override one track's volume by file name without extension (e.g. n-whoosh=0.08).
import re, sys, subprocess, os
import numpy as np
args = sys.argv[1:]; check = None
if "--check" in args: i = args.index("--check"); check = args[i + 1]; del args[i:i + 2]
html, out = args[:2]; over = dict(a.split("=") for a in args[2:]); pub = os.path.dirname(os.path.abspath(html))
tags = []
for t in re.findall(r"<audio\b[^>]*>", open(html).read()):
    a = dict(re.findall(r'([\w-]+)="([^"]*)"', t))
    if "src" in a: tags.append((a["src"], float(a.get("data-start", 0)), float(a.get("data-duration", 0)), float(a.get("data-volume", 1))))
dur = max(s + d for _, s, d, _ in tags); cmd = ["ffmpeg", "-v", "error", "-y"]; fc = []
for i, (src, st, d, v) in enumerate(tags):
    v = float(over.get(os.path.basename(src).rsplit(".", 1)[0], v)); cmd += ["-i", os.path.join(pub, src)]
    fc.append(f"[{i}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,atrim=0:{d},adelay={round(st * 1000)}:all=1,volume={v}[a{i}]")
fc.append("".join(f"[a{i}]" for i in range(len(tags))) + f"amix=inputs={len(tags)}:normalize=0:duration=longest,apad,atrim=0:{dur:.4f}[m]")
subprocess.run(cmd + ["-filter_complex", ";".join(fc), "-map", "[m]", "-c:a", "pcm_s24le", out], check=True)
print(f"{len(tags)} tracks, {dur:.2f} s -> {out}")
if check:
    dec = lambda f: np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", f, "-vn", "-ac", "2", "-ar", "48000", "-f", "f32le", "-"], capture_output=True).stdout, np.float32).reshape(-1, 2)[:, 0]
    a, b = dec(check), dec(out); n = min(len(a), len(b)) - 48000; db = lambda x: 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)
    print(f"render {db(a[:n]):.1f} dB, remix {db(b[:n]):.1f} dB, residual {db(a[:n] - b[:n]):.1f} dB" + ("" if over else "  (same mix when the residual is 25 dB or more under the signal)"))
