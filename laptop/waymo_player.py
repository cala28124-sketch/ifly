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
 
    Waymo images are 3:2; FRAME_SIZE is likely 4:3 (e.g. 160x120). Resizing
    directly would squash the image and slightly distort how fast things
    appear to expand, which matters for a looming detector. So we center-crop
    to FRAME_SIZE's aspect ratio first, then resize -- same idea as "crop to
    fit" rather than "stretch to fit".
    """
    df = pd.read_parquet(f"{camera_image_dir}/{segment_id}.parquet")
    front = df[df["key.camera_name"] == FRONT_CAMERA].sort_values(
        "key.frame_timestamp_micros"
    )
 
    target_w, target_h = FRAME_SIZE
    target_ratio = target_w / target_h
 
    for _, row in front.iterrows():
        jpg_bytes = row["[CameraImageComponent].image"]
        img = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
 
        h, w = img.shape
        current_ratio = w / h
        if current_ratio > target_ratio:
            # image is wider than target: crop the sides
            new_w = int(h * target_ratio)
            x0 = (w - new_w) // 2
            img = img[:, x0:x0 + new_w]
        elif current_ratio < target_ratio:
            # image is taller than target: crop top/bottom
            new_h = int(w / target_ratio)
            y0 = (h - new_h) // 2
            img = img[y0:y0 + new_h, :]
 
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
 
 
VEHICLE_TYPE = 1  # Waymo's Label.Type code for TYPE_VEHICLE (confirmed against label.proto)
 
 
def compute_ground_truth(boxes_df, growth_threshold=1.5, min_box_area=400,
                          smoothing_window=3, sustained_frames=2):
    """
    Ground truth for 'a vehicle is approaching fast'.
 
    boxes_df: output of load_segment_boxes(), indexed by frame timestamp.
    Tracks each vehicle's box area (size.x * size.y) across frames using
    key.camera_object_id, which stays the same for one physical object
    across the whole segment. A sustained rise in area between consecutive
    appearances of the same object means its box is expanding -- i.e. it's
    closing the distance.
 
    Two things reduce false positives from camera jitter (e.g. a road bump
    shaking the frame): small/distant boxes are the noisiest, since a couple
    pixels of jitter is a huge ratio change on a tiny box; and a single-frame
    spike is more likely noise than a real approach, so growth has to hold
    for a few frames in a row, not just one.
 
    min_box_area: ignore boxes smaller than this many pixels^2 (distant,
    noisy detections).
    smoothing_window: average area over this many frames before comparing,
    to smooth out single-frame jitter.
    sustained_frames: require growth on this many consecutive smoothed
    comparisons before counting it as a real approach.
 
    Returns: dict {timestamp: True/False}, one entry per frame timestamp
    that appears in boxes_df, True if any vehicle was sustained-approaching
    as of that frame.
    """
    vehicles = boxes_df[boxes_df["[CameraBoxComponent].type"] == VEHICLE_TYPE].copy()
    vehicles["area"] = (
        vehicles["[CameraBoxComponent].box.size.x"]
        * vehicles["[CameraBoxComponent].box.size.y"]
    )
    vehicles = vehicles[vehicles["area"] >= min_box_area]
 
    # reset_index so we can sort by (object, timestamp) instead of just timestamp
    vehicles = vehicles.reset_index()  # brings key.frame_timestamp_micros back as a column
 
    approaching_timestamps = set()
 
    for object_id, track in vehicles.groupby("key.camera_object_id"):
        track = track.sort_values("key.frame_timestamp_micros")
        # smooth with a rolling mean to absorb single-frame jitter from bumps
        smoothed = track["area"].rolling(smoothing_window, min_periods=1).mean().to_numpy()
        timestamps = track["key.frame_timestamp_micros"].to_numpy()
 
        consecutive_growth = 0
        for i in range(1, len(smoothed)):
            if smoothed[i - 1] <= 0:
                consecutive_growth = 0
                continue
            growth = smoothed[i] / smoothed[i - 1]
            if growth >= growth_threshold:
                consecutive_growth += 1
            else:
                consecutive_growth = 0
 
            if consecutive_growth >= sustained_frames:
                approaching_timestamps.add(timestamps[i])
 
    all_timestamps = boxes_df.index.unique()
    return {ts: (ts in approaching_timestamps) for ts in all_timestamps}
 
 
WAYMO_FPS = 10  # Waymo camera frames arrive at ~10 per second
WAYMO_FRAME_DELAY_MS = int(1000 / WAYMO_FPS)
 
 
def feed_to_pipeline(image_segment_id, box_segment_id=None):
    """
    Standalone smoke test: play one segment through an OpenCV window,
    same as pointing the webcam at your face. Run this file directly
    to confirm the Parquet reading works before wiring it into receiver.py.
 
    Plays back at Waymo's actual ~10fps (not 30) so the motion you see here
    matches what the eyes/brain will actually be fed once the runner
    accounts for the real frame rate -- otherwise everything looks 3x
    faster than it really is, which is exactly the frame-rate mismatch
    the team is fixing on the eyes/brain side too.
 
    If box_segment_id is given, overlays "GT: APPROACHING" whenever the
    ground truth says a vehicle's box grew fast at that timestamp -- lets
    you eyeball whether the ground truth logic looks right before trusting
    it in score_waymo.py.
    """
    ground_truth = {}
    if box_segment_id:
        boxes = load_segment_boxes(box_segment_id)
        ground_truth = compute_ground_truth(boxes)
 
    for ts, frame in load_segment_frames(image_segment_id):
        img = cv2.resize(frame, (640, 480))
        if ground_truth.get(ts):
            cv2.putText(img, "GT: APPROACHING", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow("waymo playback", img)
        if cv2.waitKey(WAYMO_FRAME_DELAY_MS) == 27:  # Esc quits
            break
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    import sys
 
    if len(sys.argv) < 2:
        print("usage: python -m laptop.waymo_player IMAGE_SEGMENT_ID [BOX_SEGMENT_ID]")
        sys.exit(1)
    box_id = sys.argv[2] if len(sys.argv) > 2 else None
    feed_to_pipeline(sys.argv[1], box_id)
 