"""Feeds a downloaded Waymo segment into the receiver, the same way the Pi sends camera frames.

Start the receiver first with Waymo settings, then run this:
    python -m laptop.receiver --fps 10 --chunk 2 --log waymo/results/SEGMENT.csv
    python -m laptop.waymo_sender SEGMENT              (as fast as the receiver keeps up; for scoring)
    python -m laptop.waymo_sender SEGMENT --realtime   (at Waymo's real 10 fps; for watching / demos)
Each frame is sent with its Waymo timestamp as its name, which the receiver writes to the log.
The fly's timing comes from --fps, not from how fast frames arrive, so sending faster is fine.
"""
import argparse
import time
import cv2
import imagezmq
from config import FRAME_PORT
from laptop.waymo_player import load_segment_frames, WAYMO_FPS

parser = argparse.ArgumentParser(description="Send a Waymo segment's front-camera frames to the receiver.")
parser.add_argument("segment", help="segment name (file waymo/camera_image/SEGMENT.parquet)")
parser.add_argument("--realtime", action="store_true", help=f"send at {WAYMO_FPS} fps instead of as fast as possible")
args = parser.parse_args()

sender = imagezmq.ImageSender(connect_to=f"tcp://127.0.0.1:{FRAME_PORT}")   # the receiver on this laptop
start, sent = time.time(), 0
for timestamp, frame in load_segment_frames(args.segment):
    if args.realtime:
        time.sleep(max(0, start + sent / WAYMO_FPS - time.time()))
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])   # same quality as the Pi
    sender.send_jpg(str(timestamp), jpg)   # waits until the receiver has taken it
    sent += 1
sender.send_jpg("__end__", jpg)   # tells the receiver the clip is over, so it closes its video/log cleanly
print(f"sent {sent} frames ({sent / WAYMO_FPS:.1f} s of driving) in {time.time() - start:.1f} s")
