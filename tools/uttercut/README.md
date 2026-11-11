# uttercut — cut talking-head takes at the utterance level

For raw takes where the speaker reads off a script: pauses before lines, false starts,
restarts, and a spoken "cut" before a retake. Whole-take transcription merges a false
start into its retake and its word timestamps drift around stutters, so word-window
cutting both leaks restarts and clips sentence endings. Splitting the take into
utterances first makes every restart its own visible line.

| Script | What it does |
|---|---|
| `utter.py <take>...` | Split each take at pauses ≥ 0.30 s (RMS envelope, thresholds relative to the take's p90 level) and transcribe every utterance on its own (faster-whisper medium, word timestamps, a prompt that keeps "cut" / "um" in the text). Writes `utt/<take>.json`. |
| `cut.py <part>` | Build `<part>/mezz.mov` from the utterances listed in `plan.json` (or `$PLAN`), butted together with no pauses. Pieces keep a 40 ms lead and their natural tail, so sentences finish. Starts are floored and ends ceiled to the frame grid. Writes `<part>/edl.json`. `DRY=1` writes only the EDL. |
| `tcheck.py <part>...` | Compare what the cut should say (source words inside each EDL piece) with a fresh transcript of the cut (`<part>/words.json`). Lists missing or added words with their output times. |
| `headchk.py <src> <t>...` | Transcribe 1.8 s from each start time. Use it to prove a piece's first word survives, or to find a clean edge by trying a few nearby times. |

## Plan format

```json
{ "part": { "take": "take-03", "vf": "optional ffmpeg crop/scale", "keep": [
    0, 3,
    {"u": 17, "from": "and most of these"},
    {"u": 11, "to": "ask me"},
    {"u": 49, "start": 134.93},
    {"u": 127, "end": 360.97} ] } }
```

- `int`: the whole utterance.
- `from`: start at the last occurrence of that phrase in the utterance. This drops a restart.
- `to`: end after its first occurrence.
- `start` / `end`: absolute source seconds.

## Workflow

1. Run `utter.py` on every take.
2. Read the utterance texts, then pick one clean reading of each script line:
   - Skip anything followed by a spoken "cut".
   - Skip a line that is read again later; take the later read.
3. Run `DRY=1 cut.py`, then `cut.py`.
4. Transcribe the cut and run `tcheck.py`.
5. Check every flagged head or tail with `headchk.py`. Transcribers drop or mishear fast,
   quiet words such as "you get", "it was… okay", or a trailing "so…".
6. Fix an edge with `start` / `end`, and re-cut.
