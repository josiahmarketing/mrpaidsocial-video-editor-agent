# bgmatte — background blur / background swap for talking-head footage (CPU, no GPU)

Person matting with MODNet (ONNX, ~26 MB) on onnxruntime, temporal smoothing of the matte, and an
OpenCV composite. No torch, no GitHub access needed.

Setup (once): `pip install onnxruntime opencv-python-headless numpy` and download the model next to the
scripts (or point `MODNET=/path/model.onnx` at it):

    curl -sSL -o tools/bgmatte/modnet.onnx https://huggingface.co/Xenova/modnet/resolve/main/onnx/model.onnx

The model file is gitignored (it is a binary download, not a source file).

- `blurbg.py in.mov out.mp4 [ss dur]` — landscape 1920x1080 @30: keeps the speaker sharp, blurs the room
  (downscaled Gaussian plate). Matte at 640x352 (env MW/MH), computed every 2nd frame. ~4 fps on 4 cores
  alone, ~1–3 fps while other CPU work runs: budget it like a render and run it in the background.
- `bgswap.py in.mov out.mp4 bg.png` — vertical 1080x1920 @30 "green screen": the speaker over a heavily
  blurred, darkened, slowly drifting version of `bg.png` (e.g. a text thread, so names stay unreadable).
  Tighter matte + edge decontamination (semi-transparent edge pixels are pulled toward the background
  tone, which removes most of the bright-wall halo around curly hair).

- `cutout.py in.mp4 out_prefix`: vertical 1080x1920 @30 "sticker cutout" for HTML compositions.
  It writes two plain videos that composite with CSS blend modes, so no alpha channel is needed:
  - `<prefix>-mk.mp4` is the multiply layer. It is white outside the sticker and black inside, with a soft
    drop shadow around it.
  - `<prefix>-fg.mp4` is the add layer: the speaker premultiplied on black, plus a white sticker outline.

  Stack them as siblings of the background graphics:
  `<video style="mix-blend-mode:multiply">` then `<video style="mix-blend-mode:plus-lighter">`.
  Anything placed before them in the DOM sits behind the speaker (oversized type tucked behind the head);
  anything after sits in front.
  - Transform both videos with the same tween.
  - Never wrap them in a transformed or faded container. That isolates the group and the blend sees no
    backdrop.
  - Env: `OUTLINE` px (default 12, 0 = none), `SHADOW` 0..1, `EVERY` (matte every Nth frame).
  - Render with JPEG frame extraction. Two full-frame videos as PNG frames fill a small disk.

All three write video only (all-intra x264); mux the cleaned audio from the source/mezzanine afterwards.
Verify on a frame crop of the hair edge, not on a full-frame thumbnail.
