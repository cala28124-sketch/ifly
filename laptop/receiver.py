# Frames arrive in color (COLOR_SIZE) and are turned into 160x120 grayscale here for the fly eyes.
# The vehicle check runs once per chunk on the last color frame (it needs color); for now its
# boxes are only shown in the viewer and don't affect the warning.
#
# Live:   python -m laptop.receiver
# Waymo:  python -m laptop.receiver --fps 10 --chunk 2 --log waymo/results/SEGMENT.csv
#         (then feed it with: python -m laptop.waymo_sender SEGMENT)
# Demo video: add --record demo.mp4 (camera view + circuit panel + brain map, every frame, real speed);
#         --no-window records or logs without opening windows.
import argparse, csv, cv2, zmq, imagezmq, numpy as np
from config import (FRAME_PORT, COMMAND_PORT, CHUNK, FAKE_EYES, EYES_MODEL, FAKE_BRAIN, FRAME_SIZE, EYES_ZOOM,
                    VEHICLE_CHECK, CAMERA_FPS)
from laptop.eyes import FlyEyes
from laptop.brain import FlyBrainModel
from laptop.viewer import camera_view, compose, display, recording_frame
if VEHICLE_CHECK:
    from vehicle.detector import detect_vehicles

parser = argparse.ArgumentParser(description="Laptop side: frames in, fly eyes + brain, warnings out.")
parser.add_argument("--fps", type=float, default=CAMERA_FPS, help="frame rate of the incoming frames (Waymo: 10)")
parser.add_argument("--chunk", type=int, default=CHUNK, help="frames per decision (Waymo: 2, about 0.2 s)")
parser.add_argument("--log", help="write every decision to this CSV file (for scoring)")
parser.add_argument("--record", help="save a video of the camera view, circuit panel and brain map (.mp4)")
parser.add_argument("--no-window", action="store_true", help="don't open windows (recording / scoring only)")
args = parser.parse_args()

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

eyes = FlyEyes(fake=FAKE_EYES, model=EYES_MODEL, fps=args.fps)
brain = FlyBrainModel(fake=FAKE_BRAIN, fps=args.fps, chunk=args.chunk)
chunk = []
log = None
if args.log:   # one row per decision; the frame's name is the Pi's hostname live, or the Waymo timestamp
    log_file = open(args.log, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["frame", "loom_left", "loom_right", "escape", "turn", "fired", "vehicles", "turning"])

video, last = None, None   # the video writer, and the latest decision's (seen, thought, vehicles, view)

def record(frame):
    """Write this frame to the video with the latest decision's overlays, panel and brain map."""
    global video
    seen, thought, vehicles, view = last
    picture = recording_frame(dict(view, camera=camera_view(frame, seen, thought, vehicles)))
    if video is None:
        h, w = picture.shape[:2]
        for codec in ("avc1", "mp4v"):   # H.264 plays everywhere; mp4v is the fallback
            video = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*codec), args.fps, (w, h))
            if video.isOpened():
                break
    video.write(picture)

try:
    while True:
        name, jpg = hub.recv_jpg()
        hub.send_reply(b"OK")
        if name == "__end__":   # waymo_sender's end-of-clip signal (the Pi never sends it)
            break
        color = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        frame = cv2.resize(zoom_center(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), EYES_ZOOM), FRAME_SIZE)   # gray 160x120 for the eyes
        chunk.append(frame)
        if len(chunk) < args.chunk:
            if args.record and last:
                record(frame)
            continue
        seen = eyes.step(chunk)
        chunk = []
        thought = brain.step(seen["loom_left"], seen["loom_right"])
        turn = threat_side(seen, thought)
        pub.send_json({"turn": turn, "escape": thought["escape"]})
        vehicles = None
        if VEHICLE_CHECK:   # once per chunk (~30 ms), on the full color frame
            vehicles = detect_vehicles(color)[1]
            for v in vehicles:
                v["view_box"] = to_eye_view(v["box"], color.shape, EYES_ZOOM)
        if log:
            log.writerow([name, f"{seen['loom_left']:.3f}", f"{seen['loom_right']:.3f}", int(thought["escape"]),
                          f"{turn:.2f}", " > ".join(thought["fired"]),
                          ";".join(f"{v['type']}:{v['confidence']:.2f}" for v in vehicles or []),
                          int(seen.get("turning", False))])
            log_file.flush()   # so the file is complete even if the receiver is stopped with Ctrl+C
        view = compose(frame, seen, thought, vehicles)   # once per decision (the brain map keeps a memory)
        last = (seen, thought, vehicles, view)
        if args.record:
            record(frame)
        if not args.no_window and display(view):
            break
except KeyboardInterrupt:
    pass
finally:   # close the files properly so the video plays and the log is complete
    if video is not None:
        video.release()
        print(f"saved {args.record}")
    if log:
        log_file.close()
