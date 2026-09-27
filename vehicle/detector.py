"""Vehicle check: finds cars, motorcycles, buses and trucks in a color frame with YOLO (yolo11n,
pretrained on COCO; runs ~30 ms per 320x240 frame on a laptop CPU). Needs color: on small grayscale
frames it misses even large vehicles. Needs the ultralytics package (uv pip install ultralytics).
"""
import pathlib
from ultralytics import YOLO


# Load YOLO once when this module is imported (weights live next to this file)
model = YOLO(pathlib.Path(__file__).parent / "yolo11n.pt")


# YOLO class IDs that we consider vehicles
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


def detect_vehicles(frame):

    # Give the frame to YOLO
    results = model(frame, verbose=False)

    # Assume no vehicle until one is found
    vehicle_present = 0

    # Store information about detected vehicles
    detections = []


    # Look through everything YOLO detected
    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])


            # Only care about vehicles
            if class_id in VEHICLE_CLASSES:

                vehicle_present = 1

                # Get bounding box
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                detections.append({
                    "type": VEHICLE_CLASSES[class_id],
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2]
                })


    return vehicle_present, detections
