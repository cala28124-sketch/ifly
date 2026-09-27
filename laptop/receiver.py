# Frames arrive in color (COLOR_SIZE) and are turned into 160x120 grayscale here for the fly eyes.
# The vehicle check runs once per chunk on the last color frame (it needs color); for now its
# boxes are only shown in the viewer and don't affect the warning.
import cv2, zmq, imagezmq, numpy as np
from config import (FRAME_PORT, COMMAND_PORT, CHUNK, FAKE_EYES, EYES_MODEL, FAKE_BRAIN, FRAME_SIZE, EYES_ZOOM,
                    VEHICLE_CHECK, CAMERA_FPS)
from laptop.eyes import FlyEyes
from laptop.brain import FlyBrainModel
from laptop.viewer import show
if VEHICLE_CHECK:
    from vehicle.detector import detect_vehicles

hub = imagezmq.ImageHub(open_port=f"tcp://*:{FRAME_PORT}")
pub = zmq.Context().socket(zmq.PUB)
pub.bind(f"tcp://*:{COMMAND_PORT}")
def zoom_center(image, zoom):
    """Keep the centre 1/zoom of the image (zoom 1 = unchanged)."""
    h, w = image.shape[:2]
    ch, cw = int(h / zoom), int(w / zoom)
    return image[(h - ch) // 2:(h + ch) // 2, (w - cw) // 2:(w + cw) // 2]

def to_eye_view(box, color_shape, zoom):
    """A box in color-frame pixels -> the same box in the eyes' zoomed 160x120 view."""
    h, w = color_shape[:2]
    ch, cw = int(h / zoom), int(w / zoom)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    sx, sy = FRAME_SIZE[0] / cw, FRAME_SIZE[1] / ch
    x1, y1, x2, y2 = box
    return [(x1 - x0) * sx, (y1 - y0) * sy, (x2 - x0) * sx, (y2 - y0) * sy]

def threat_side(seen, thought):
    """-1 = threat on the image's left, +1 = right, 0 = both. Taken from which giant fiber (DNp01) fired
    when the real brain runs, otherwise from the loom scores."""
    left, right = thought.get("activity", {}).get("DNp01", (seen["loom_left"], seen["loom_right"]))
    return 0.0 if left + right == 0 else (right - left) / (right + left)

eyes = FlyEyes(fake=FAKE_EYES, model=EYES_MODEL, fps=CAMERA_FPS)
brain = FlyBrainModel(fake=FAKE_BRAIN, fps=CAMERA_FPS, chunk=CHUNK)
chunk = []

while True:
    _, jpg = hub.recv_jpg()
    hub.send_reply(b"OK")
    color = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    frame = cv2.resize(zoom_center(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), EYES_ZOOM), FRAME_SIZE)   # gray 160x120 for the eyes
    chunk.append(frame)
    if len(chunk) < CHUNK:
        continue
    seen = eyes.step(chunk)
    chunk = []
    thought = brain.step(seen["loom_left"], seen["loom_right"])
    pub.send_json({"turn": threat_side(seen, thought), "escape": thought["escape"]})
    vehicles = None
    if VEHICLE_CHECK:   # once per chunk (~30 ms), on the full color frame
        vehicles = detect_vehicles(color)[1]
        for v in vehicles:
            v["view_box"] = to_eye_view(v["box"], color.shape, EYES_ZOOM)
    if show(frame, seen, thought, vehicles):
        break