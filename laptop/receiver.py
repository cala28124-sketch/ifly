import cv2, zmq, imagezmq, numpy as np
from config import FRAME_PORT, COMMAND_PORT, CHUNK, FAKE_EYES, EYES_MODEL, FAKE_BRAIN
from laptop.eyes import FlyEyes
from laptop.brain import FlyBrainModel
from laptop.viewer import show

hub = imagezmq.ImageHub(open_port=f"tcp://*:{FRAME_PORT}")
pub = zmq.Context().socket(zmq.PUB)
pub.bind(f"tcp://*:{COMMAND_PORT}")
eyes, brain, chunk = FlyEyes(fake=FAKE_EYES, model=EYES_MODEL), FlyBrainModel(fake=FAKE_BRAIN), []

while True:
    _, jpg = hub.recv_jpg()
    hub.send_reply(b"OK")
    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)
    chunk.append(frame)
    if len(chunk) < CHUNK:
        continue
    seen = eyes.step(chunk)
    chunk = []
    thought = brain.step(seen["loom_left"], seen["loom_right"])
    pub.send_json({"turn": 0.0, "escape": thought["escape"]})
    if show(frame, seen, thought):
        break