# gaze.py <video> <out.json> — per-frame blink / look-down / look-side scores (MediaPipe face blendshapes) for a talking head.
# Use it on a cut to list stretches where the speaker's eyes are off the lens (reading the script): rows are
# [frame, blink, down, side, up]; blink and down both >= 0.45, or side >= 0.5, for 4+ frames is a glance worth hiding.
# Needs: pip install mediapipe, the system library libegl1, and the model file (FACE_MODEL env):
#   https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
import sys, json, subprocess, numpy as np, mediapipe as mp, os
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
opt = vision.FaceLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=os.environ.get("FACE_MODEL", os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "face_landmarker.task"))), output_face_blendshapes=True, num_faces=1)
det = vision.FaceLandmarker.create_from_options(opt)
W, H = 540, 960
p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", sys.argv[1], "-vf", f"scale={W}:{H}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
rows = []; n = 0
while True:
    b = p.stdout.read(W * H * 3)
    if len(b) < W * H * 3: break
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.frombuffer(b, np.uint8).reshape(H, W, 3))))
    if r.face_blendshapes:
        d = {c.category_name: c.score for c in r.face_blendshapes[0]}
        rows.append([n, round((d["eyeBlinkLeft"] + d["eyeBlinkRight"]) / 2, 3), round((d["eyeLookDownLeft"] + d["eyeLookDownRight"]) / 2, 3),
                     round(max(d["eyeLookOutLeft"], d["eyeLookInRight"], d["eyeLookInLeft"], d["eyeLookOutRight"]), 3), round((d["eyeLookUpLeft"] + d["eyeLookUpRight"]) / 2, 3)])
    else: rows.append([n, -1, -1, -1, -1])
    n += 1
json.dump(rows, open(sys.argv[2], "w"))
a = np.array(rows); print("frames", n, "no-face", int((a[:, 1] < 0).sum()))
for k, name in enumerate(["blink", "down", "side", "up"], 1): print(name, "p10/p50/p90/p98", [round(float(np.percentile(a[:, k], q)), 2) for q in (10, 50, 90, 98)])
