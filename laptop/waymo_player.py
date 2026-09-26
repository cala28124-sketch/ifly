"""
Reads Waymo v2 Parquet segments and yields frames in the same shape the
webcam pipeline expects: 160x120 grayscale uint8 NumPy arrays.

Also pulls matching camera_box ground truth so a later scoring script
(laptop/score_waymo.py) can check "did we warn before the box grew fast".
"""

import pandas as pd
import numpy as np
import cv2

from config import FRAME_SIZE

FRONT_CAMERA = 1  # Waymo's camera_name code for the front-facing camera


def load_segment_frames(segment_id, camera_image_dir="waymo/camera_image"):
    """
    Yields (timestamp, frame) for the front camera of one segment,
    frame already resized to FRAME_SIZE and grayscale.
    """
    df = pd.read_parquet(f"{camera_image_dir}/{segment_id}.parquet")
    front = df[df["key.camera_name"] == FRONT_CAMERA].sort_values(
        "key.frame_timestamp_micros"
    )

    for _, row in front.iterrows():
        jpg_bytes = row["[CameraImageComponent].image"]
        img = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        frame = cv2.resize(img, FRAME_SIZE)
        yield row["key.frame_timestamp_micros"], frame


def load_segment_boxes(segment_id, camera_box_dir="waymo/camera_box"):
    """
    Returns the camera_box dataframe for one segment, front camera only,
    indexed by timestamp so it's easy to look up ground truth per frame.
    Exact column names: confirm against the v2 tutorial notebook, since
    Waymo's schema has shifted between releases.
    """
    df = pd.read_parquet(f"{camera_box_dir}/{segment_id}.parquet")
    front = df[df["key.camera_name"] == FRONT_CAMERA]
    return front.set_index("key.frame_timestamp_micros")


def feed_to_pipeline(segment_id):
    """
    Standalone smoke test: play one segment through an OpenCV window,
    same as pointing the webcam at your face. Run this file directly
    to confirm the Parquet reading works before wiring it into receiver.py.
    """
    for ts, frame in load_segment_frames(segment_id):
        cv2.imshow("waymo playback", cv2.resize(frame, (640, 480)))
        if cv2.waitKey(33) == 27:  # ~30fps playback, Esc quits
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m laptop.waymo_player SEGMENT_ID")
        sys.exit(1)
    feed_to_pipeline(sys.argv[1])