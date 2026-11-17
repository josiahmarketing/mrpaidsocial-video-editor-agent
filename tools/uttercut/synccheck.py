# synccheck.py <part> <audio-or-video-file> — is every piece's voice where its picture is?
# Reads <part>/edl.json and the take's audio (AUDIO env = the cleaned voice the cut was made from, else the take itself),
# rebuilds each piece's voice, finds it in the given file by correlation and prints how far it sits from the piece's picture
# position. Run it on the cut's own audio AND on the finished master: the second run also catches a stale voice file,
# an offset added by the renderer, or a re-timed edit. Exit code 1 if any piece is more than TOL ms out (default 15).
#   --fix <out.wav>   instead of checking, write the cut's audio rebuilt sample-exactly (repairs a cut made before pieces.py)
#   --remap <words.json> <old-audio> <out.json>   move word times measured on a drifted audio onto the exact timeline
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pieces

k = sys.argv[1]; edl = json.load(open(f"{k}/edl.json")); src = os.environ.get("AUDIO") or edl["src"]
pre = None if os.environ.get("AUDIO") else os.environ.get("PRE")
if sys.argv[2] == "--fix":
    pcm = pieces.build(edl, src, pre); pieces.write_wav(sys.argv[3], pcm)
    print(f"{sys.argv[3]}: {len(pcm) / 48000:.3f} s, picture {edl['segments'][-1]['out_end']:.3f} s"); sys.exit()
if sys.argv[2] == "--remap":
    words, old, out = sys.argv[3:6]; off = pieces.offsets(edl, src, old, pre); last = 0.0; starts = []
    for i, t, ms, c in off:   # a piece that could not be measured (too short, too quiet) takes its neighbour's offset
        if ms is not None and c > 0.6: last = ms
        starts.append((t + last / 1000, last / 1000))
    W = json.load(open(words))
    for w in W:
        d = next((d for s, d in reversed(starts) if w["start"] >= s - 0.02), starts[0][1])
        w["start"] = round(w["start"] - d, 3); w["end"] = round(w["end"] - d, 3)
    json.dump(W, open(out, "w")); print(f"{out}: {len(W)} words moved by {-starts[0][1] * 1000:.0f} .. {-starts[-1][1] * 1000:.0f} ms"); sys.exit()
tol = float(os.environ.get("TOL", 15)); res = pieces.offsets(edl, src, sys.argv[2], pre); bad = 0
for i, t, ms, c in res:
    flag = "" if ms is None or abs(ms) <= tol else "  <-- OUT OF SYNC"
    if ms is not None and c > 0.5 and abs(ms) > tol: bad += 1
    if flag or i % max(1, len(res) // 12) == 0 or i == len(res) - 1: print(f"piece {i:3d}  at {t:7.2f} s  voice {'  n/a ' if ms is None else f'{ms:+7.1f} ms'}  (match {c:.2f}){flag}")
m = [ms for _, _, ms, c in res if ms is not None and c > 0.5]
print(f"{len(m)} of {len(res)} pieces measured: voice vs picture {min(m):+.1f} .. {max(m):+.1f} ms; {bad} outside +-{tol:.0f} ms")
sys.exit(1 if bad else 0)
