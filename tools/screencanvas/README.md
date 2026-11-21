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
| `mkcanvas.py <part>` | The screen side of a cut made with `uttercut`: reads `<part>/edl.json`, decodes the recording once and writes `<part>/public/canvas.mp4` with the same frame count as the cut, joins dissolved. `OFFSET` = seconds the screen recording runs ahead of the camera. `shifts.json` (`{"piece": seconds}`) slides the recording under a piece. `regions.json` (`{"piece": [[x0, y0, x1, y1], ...]}`) shows plate rectangles as the clean plate during a piece. |

## Workflow

1. **Plate.** Find a stretch where the whole drawing is on screen with no toolbar (often the first seconds, before markup
   mode). Take the median of a few frames and crop to the paper: `plate.png`. Then look at its outer 20 rows and columns
   for anything dark: the lower edge of an app button or a scroll indicator survives the crop, and lifting the shade under
   the top bar makes it darker. It is invisible while the edit is zoomed in and shows as a speck the moment the whole
   page comes on screen. Paint it out of the plate before rendering the canvas.
2. **Check the audio.** A tablet screen recording is often SILENT; the voice is on the camera file. Probe it before
   planning the cut.
3. **Line up with the camera.** Two recordings started "together" by hand are easily a second apart. If the speaker made a
   sync mark (held the tablet up to the camera and scribbled before starting), find the first frame with that ink in
   each file: the difference is `OFFSET`, to a frame. A screen recording has a variable frame rate (frames exist only
   when the screen changes), so read frame TIMES (`-vf showinfo`, `pts_time`), never frame counts. Without a mark,
   compare `track.json` stroke times with the phrases in the camera transcript. A lavalier with a noise gate records no
   pencil sound, so sound cannot do it; phrases and strokes can, to a few tenths of a second, which is enough when the
   hand is not in the camera frame.
4. **Cut the camera take** with `uttercut` as usual, then run `mkcanvas.py`.
5. **Shift strokes under the words.** People draw the underline just after saying the line, in the pause the cut removes.
   For each piece compare the stroke time with the piece's source window and add a shift so the stroke plays while the line
   is spoken. A piece that ends mid-stroke gets a shift too.
6. **Match the white.** A recorded "white" page is about 251,253,250, and it is shaded by 10-20 levels under the app's top
   bar and near the edge of the screen when the speaker has zoomed in. Take the bar's shade out of the plate (divide its
   top rows by their own paper level), set `CHROME` to the bar plus its shade, and lift the video
   (`filter: brightness(1.1)`) so its paper meets pure white with no visible edge. Clip a few pixels off the video's edges.
7. **Strokes the speaker's own zoom cuts in half.** Where the speaker zoomed in for a later section, the canvas shows the
   plate outside their view, so a stroke left over from an earlier section ends at a straight line. List that rectangle
   for those pieces in `regions.json`. A piece with no drawing in it can be the plate itself (`[[0, 0, W, H]]`).

## Revealing the drawing in step with the script

A finished drawing that is on screen from the first second gives the viewer the whole answer at once. Because the canvas
is steady in plate coordinates, the edit can hold parts back and bring them in when they are spoken:

- **Curtains.** White boxes in plate coordinates over the parts not reached yet, at about 0.89 opacity: a faint ghost of
  what is coming stays visible, so the page never looks empty. Fully opaque curtains left a blank lower half.
- **Entrances from the same plate.** Cut each element out of the plate (white to transparent), animate that piece in at
  its own position, and remove curtain and piece together when it has landed: the recording underneath shows the same
  pixels, so there is no jump, and the speaker's live strokes appear on it afterwards. Give the pieces the same
  brightness lift as the video. Elements that touch (an arrow whose head sits on a card) are split by colour and by
  connected component, and the line they crossed is patched from its neighbours.
- **Know where the live strokes are.** Before placing a curtain, difference the canvas against the plate at the end of
  every piece and list the stroke boxes. A curtain edge across a live stroke cuts it with a straight line; use a
  soft-edged veil there, or reveal that part earlier.
- **Views are sized to the lettering.** Lettering height times view scale should come to 35 px or more on a 1080-wide
  frame. A view scaled to fit two far-apart labels makes both unreadable: frame the one being talked about, leave the
  other as a ghost, and when both are needed at once lift them out of the plate and set them side by side in screen
  coordinates, with the whole page small underneath.
- **Fade the panel's top edge.** Every view cuts something at the top of the panel, and it shows as slivers beside the
  caption bar. One white band across the top (opaque to the caption's lower edge, fading over the next 25 px) hides them;
  place each view so its subject starts below the band.
- **App buttons move with the speaker's zoom.** A corner button patched out of the unzoomed plate sits on a different part
  of the page once the speaker has zoomed in. Check the canvas at each zoom state and clip or patch again.
- **Full-size graphics go in the window, not on the page.** In a split screen the safe band under the camera is about
  400 px tall. A meter or a document screenshot drawn at the page's scale comes out a third of the size it needs:
  place those in screen coordinates over the band, and dim the page behind them with a white veil.

Environment: `SRC` (recording path, default `dl/raw.mp4`), `FRAME` (its size, default `2732x2048`), `MARKUP_FROM` (second at
which the app's bars appear, default 0), `CHROME` (height of the top bars in px, default 196), `PATCH` (one plate rectangle
to paint white), `OFFSET` (seconds the recording runs ahead of the camera, default 0), `SHIFTS` / `REGIONS` (the per-piece files,
default `shifts.json` / `regions.json`: a revision that drops or adds a piece renumbers the pieces, so give it its own
files). Run from the project folder that holds `plate.png`.
