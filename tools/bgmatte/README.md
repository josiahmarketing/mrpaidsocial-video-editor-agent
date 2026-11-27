# bgmatte — background blur / background swap for talking-head footage (CPU, no GPU)

Person matting with MODNet (ONNX, ~26 MB) on onnxruntime, temporal smoothing of the matte, and an
OpenCV composite. No torch, no GitHub access needed.

Setup (once): `pip install onnxruntime opencv-python-headless numpy` and download the model next to the
scripts (or point `MODNET=/path/model.onnx` at it):

    curl -sSL -o tools/bgmatte/modnet.onnx https://huggingface.co/Xenova/modnet/resolve/main/onnx/model.onnx

The model file is gitignored (it is a binary download, not a source file).

- `blurbg.py in.mov out.mp4 [ss dur]` — @30, at the input's own size unless `SIZE=WxH` is set (it then scales):
  keeps the speaker sharp, blurs the room (downscaled Gaussian plate). Matte 640x352 for landscape, 352x640 for
  portrait (env MW/MH), computed every 2nd frame. (It used to default to 1920x1080 without scaling, so a 9:16 input
  came out as scrambled landscape frames while the run reported success: probe the output's size before deleting
  the input.) ~4 fps on 4 cores
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
  - Env: `OUTLINE` px (default 12, 0 = none), `SHADOW` 0..1, `EVERY` (matte every Nth frame),
    `MATTE="WxH"` (matte size in multiples of 32; 512x896 is cleaner around hair at about half the speed).
  - Render with JPEG frame extraction. Two full-frame videos as PNG frames fill a small disk.
  - **The multiply layer must decode as exact white outside the sticker.** If it is 2 % short, the whole
    video box dims the backdrop and shows as a rectangle as soon as the layer is moved (a drop, a slide).
    Every 8-bit RGB to YUV round trip loses 1-2 % of white, so `cutout.py` stores the matte's top end
    above nominal white. After any later re-encode of the matte (joining sections, trimming), lift it again:
    `ffmpeg -i mk.mp4 -vf "lutyuv=y='if(gt(val,212),min(212+(val-212)*1.9,252),val)'" -c:v libx264 -crf 12 out.mp4`.
    Prove it in the composition, not in the file: shift the layer, snapshot, and compare the backdrop's
    colour inside and outside the video's box. They must be equal.

  When the matte alone is not enough (it keeps furniture, or loses pale clothes), `cutout.py` takes a second
  opinion. Add only what the footage needs, and look at a strip of consecutive frames after each one:
  - `SEG=<selfie_multiclass.tflite>` (MediaPipe class segmenter: hair, skin, clothes, accessories). The matte
    is kept only near those classes (`SEG_PAD` px of slack), and clothes or skin the matte lost are filled
    back in below `BODY_FROM` (px row). `SEG_CROP="y0,y1"` runs it on those rows only. The accessories class
    is part of the gate on purpose: without it a clip-on microphone is cut out of the chest.
  - `NECK=<px row>`: above it only hair and skin count. A dark chair back is "clothing" to the segmenter.
  - `COOL="thr[,to_row[,dark]]"` with `COOL_SKIP="x0,y0,x1,y1"`: a dark chair back or headrest behind dark
    hair. Neither model separates them, so pieces of the chair come and go from frame to frame. Colour does:
    dark hair is warm (red above blue), black plastic and mesh are neutral to blue. Print the local
    red-minus-blue of a hair patch and of a chair patch first, and put the threshold between them. The skip
    box exempts a black top under the chin, which is dark and cool as well.
  - `HEAD="cx,cy,rx,ry"` or `KEEP=<mask.png>` (from `hairmask.py`): for a face-locked speaker, an arch or a
    picture of where the head may be. Good for chair wings beside the hair; it cannot remove what sits
    inside the hair's usual outline.
  - `SOLID=<gamma>`: fast hands are half transparent over a wide blur and the white sticker paper shows
    through them; 0.55 keeps them flesh-coloured.
- `hairmask.py in.mp4 out.png`: for a face-locked speaker, the share of frames in which the segmenter calls
  each pixel hair or face, thresholded (`FREQ`) and widened (`PAD`) into a mask for `KEEP`.

All three write video only (all-intra x264); mux the cleaned audio from the source/mezzanine afterwards.
Verify on a frame crop of the hair edge, not on a full-frame thumbnail.
