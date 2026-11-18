# screencanvas — a drawing-app screen recording as a clean, steady canvas

For videos where the speaker draws on a tablet while talking (a diagram prepared in advance, circles and underlines added
live). The raw screen recording is hard to use as it is: the app's bars and floating toolbar cover parts of the drawing,
the speaker pinch-zooms between takes, and each retake leaves the drawing in a different state.

These scripts fit every frame to one clean picture of the drawing (the "plate"), so the output is always in the plate's
coordinates: the speaker's own zooms disappear, the app chrome is replaced by the plate, and the live strokes stay. The
edit then does its own focus moves with one set of coordinates for the whole video.

| Script | What it does |
|---|---|
| `track.py` | Samples the recording 15 times a second: scale and offset of the drawing against the plate (ORB matches, similarity without rotation), the toolbar box, and how much ink is on screen. `track.json` shows when strokes are drawn and undone and when the view changes. Use it to line the recording up with the camera take and to see which retake a stroke belongs to. |
| `canvas.py` | `Canvas().frame(frame, markup=True)` returns one clean canvas image. `canvas.py test t1 t2 ...` writes test frames. |
| `mkcanvas.py <part>` | The screen side of a cut made with `uttercut`: reads `<part>/edl.json`, decodes the recording once and writes `<part>/public/canvas.mp4` with the same frame count as the cut, joins dissolved. `shifts.json` (`{"piece": seconds}`) slides the recording under a piece. |

## Workflow

1. **Plate.** Find a stretch where the whole drawing is on screen with no toolbar (often the first seconds, before markup
   mode). Take the median of a few frames and crop to the paper: `plate.png`.
2. **Check the audio.** A tablet screen recording is often SILENT; the voice is on the camera file. Probe it before
   planning the cut.
3. **Line up with the camera.** Compare `track.json` stroke times with the phrases in the camera transcript. A lavalier
   with a noise gate records no pencil sound, so sound cannot do it; phrases and strokes can, to a few tenths of a second,
   which is enough when the hand is not in the camera frame.
4. **Cut the camera take** with `uttercut` as usual, then run `mkcanvas.py`.
5. **Shift strokes under the words.** People draw the underline just after saying the line, in the pause the cut removes.
   For each piece compare the stroke time with the piece's source window and add a shift so the stroke plays while the line
   is spoken. A piece that ends mid-stroke gets a shift too.
6. **Match the white.** A recorded "white" page is about 251,253,250. Lift it (`filter: brightness(1.08)` on the video) so it
   meets a pure white page with no visible edge, and clip a few pixels off the video's edges.

Environment: `SRC` (recording path, default `dl/raw.mp4`), `FRAME` (its size, default `2732x2048`), `MARKUP_FROM` (second at
which the app's bars appear, default 0), `CHROME` (height of the top bars in px, default 196), `PATCH` (one plate rectangle
to paint white). Run from the project
folder that holds `plate.png`.
