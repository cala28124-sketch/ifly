LAPTOP_IP = "172.20.10.9"     # local test; on the Pi use the Mac's hotspot IP (ipconfig getifaddr en0)
FRAME_PORT = 5555           # Pi -> laptop frames (imagezmq)
COMMAND_PORT = 5556         # laptop -> Pi commands (pyzmq)
FRAME_SIZE = (160, 120)
CHUNK = 5                   # frames per eye-model step; smaller = less lag
WARN_THRESHOLD = 0.6        # tune on webcam and Waymo clips
FAKE_EYES = False           # True = frame-difference stand-in; False = flyvis fly eyes
EYES_MODEL = "flow/0000/000"  # flyvis ensemble model; needs tools/tuning/<model>.json
LED_PIN = 17                # Pi GPIO (BCM) number for the warning LED    -> physical pin 11
BUZZER_PIN = 27             # Pi GPIO (BCM) number for the warning buzzer -> physical pin 13
WARN_HOLD = 0.5             # seconds the LED/buzzer stay on after the last warning
