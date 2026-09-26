import cv2

def show(frame, seen, thought):
    img = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_GRAY2BGR)
    cv2.putText(img, f"loom L {seen['loom_left']:.2f}  R {seen['loom_right']:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if thought["escape"]:
        cv2.putText(img, "WARN: " + " > ".join(thought["fired"]),
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.imshow("fly brain", img)
    return cv2.waitKey(1) == 27   # True when Esc is pressed
