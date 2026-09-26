import cv2
import zmq # type: ignore
import imagezmq # type: ignore
import numpy as np

from config import FRAME_PORT, COMMAND_PORT, CHUNK, FAKE_EYES, EYES_MODEL

from laptop.eyes import FlyEyes
from laptop.brain import FlyBrainModel
from laptop.viewer import show

from vehicle.detector import detect_vehicles


# ---------------------------------
# Networking
# ---------------------------------

hub = imagezmq.ImageHub(
    open_port=f"tcp://*:{FRAME_PORT}"
)

pub = zmq.Context().socket(zmq.PUB)

pub.bind(
    f"tcp://*:{COMMAND_PORT}"
)


# ---------------------------------
# Fly brain
# ---------------------------------

eyes = FlyEyes(
    fake=FAKE_EYES,
    model=EYES_MODEL
)

brain = FlyBrainModel()

chunk = []


# ---------------------------------
# Main loop
# ---------------------------------

while True:

    # Receive JPEG from sender.py
    _, jpg = hub.recv_jpg()

    # Tell sender we received it
    hub.send_reply(b"OK")


    # Convert JPEG into OpenCV image
    frame = cv2.imdecode(
        np.frombuffer(jpg, np.uint8),
        cv2.IMREAD_GRAYSCALE
    )


    # ---------------------------------
    # VEHICLE DETECTION
    # ---------------------------------

    vehicle_present, detections = detect_vehicles(frame)

    print("Vehicle:", vehicle_present)


    # ---------------------------------
    # FLY EYES
    # ---------------------------------

    chunk.append(frame)

    if len(chunk) < CHUNK:
        continue

    seen = eyes.step(chunk)

    chunk = []


    # ---------------------------------
    # FLY BRAIN
    # ---------------------------------

    thought = brain.step(
        seen["loom_left"],
        seen["loom_right"]
    )


    # ---------------------------------
    # Send command
    # ---------------------------------

    pub.send_json({
        "turn": 0.0,
        "escape": thought["escape"]
    })


    # Show existing viewer
    if show(frame, seen, thought):
        break