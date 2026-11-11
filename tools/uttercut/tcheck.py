# tcheck.py <part>... — compare what the cut SHOULD say (source-utterance words inside each EDL piece)
# with what a fresh transcript of the cut DOES say (<part>/words.json). Prints words missing from the cut
# (clipped heads/tails) and words added (leaked neighbours), with the output time, so each can be checked.
import sys, os, json, re, difflib
n = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())
for k in sys.argv[1:]:
    P = json.load(open(os.environ.get("PLAN", "plan.json")))[k]; U = json.load(open(f"utt/{P['take']}.json"))["utts"]
    E = json.load(open(f"{k}/edl.json"))["segments"]; W = json.load(open(f"{k}/words.json"))
    exp = []
    for s in E:
        for u in U:
            for w in u["words"]:
                mid = (w["s"] + w["e"]) / 2
                if s["src_start"] <= mid <= s["src_end"]: exp.append((n(w["w"]), s["out_start"] + mid - s["src_start"]))
    got = [(n(w["text"]), w["start"]) for w in W]
    A, B = [x[0] for x in exp if x[0]], [x[0] for x in got if x[0]]
    ea, gb = [x for x in exp if x[0]], [x for x in got if x[0]]
    bad = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
        if op == "equal": continue
        t = ea[i1][1] if i1 < len(ea) else gb[j1][1]
        bad.append(f"  {t:6.2f}s {op}: expected '{' '.join(A[i1:i2])}' heard '{' '.join(B[j1:j2])}'")
    print(f"== {k}: {len(bad)} differences"); print("\n".join(bad))
