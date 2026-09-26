import cv2
import numpy as np

BAR_FULL = {"LC4": 200, "LPLC2": 400}   # spikes per chunk that fill a bar (strong looming)

def circuit_panel(thought, height):
    """Side panel: spikes this chunk in the escape circuit, left and right."""
    panel = np.full((height, 300, 3), 30, np.uint8)
    cv2.putText(panel, "escape circuit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(panel, "LEFT", (90, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, "RIGHT", (200, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    activity = thought.get("activity", {})
    if not activity:
        cv2.putText(panel, "(fake brain)", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        return panel
    for row, name in enumerate(["LC4", "LPLC2"]):
        y = 100 + row * 60
        cv2.putText(panel, name, (10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        for col, count in enumerate(activity[name]):
            x = 80 + col * 110
            fill = int(90 * min(count / BAR_FULL[name], 1))
            cv2.rectangle(panel, (x, y), (x + 90, y + 20), (80, 80, 80), 1)
            cv2.rectangle(panel, (x, y), (x + fill, y + 20), (0, 200, 255), -1)
            cv2.putText(panel, str(count), (x, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y = 250   # giant fiber: one neuron per side, lit when it spiked this chunk
    cv2.putText(panel, "DNp01", (10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    for col, count in enumerate(activity["DNp01"]):
        center = (125 + col * 110, y)
        cv2.circle(panel, center, 16, (0, 0, 255) if count else (80, 80, 80), -1 if count else 2)
        if count:
            cv2.putText(panel, str(count), (center[0] - 6, center[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    if thought["escape"]:
        cv2.putText(panel, "ESCAPE", (90, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return panel

def show(frame, seen, thought):
    img = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_GRAY2BGR)
    cv2.putText(img, f"loom L {seen['loom_left']:.2f}  R {seen['loom_right']:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if thought["escape"]:
        cv2.putText(img, "WARN: " + " > ".join(thought["fired"]),
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.imshow("fly brain", np.hstack([img, circuit_panel(thought, img.shape[0])]))
    return cv2.waitKey(1) == 27   # True when Esc is pressed
