import cv2, imagezmq, socket
from config import LAPTOP_IP, FRAME_PORT, FRAME_SIZE

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
    gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), FRAME_SIZE)
    ok, jpg = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 80])
    sender.send_jpg(name, jpg)
