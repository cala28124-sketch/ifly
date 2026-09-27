import cv2, imagezmq, socket
from config import LAPTOP_IP, FRAME_PORT, COLOR_SIZE

sender = imagezmq.ImageSender(connect_to=f"tcp://{LAPTOP_IP}:{FRAME_PORT}")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("camera 0 not available (check permission / index)")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
name = socket.gethostname()
while True:
    ok, frame = cap.read()
    if not ok:
        continue
    small = cv2.resize(frame, COLOR_SIZE)   # stays in color; the laptop makes the gray copy for the eyes
    ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    sender.send_jpg(name, jpg)
