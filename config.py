LAPTOP_IP = "172.20.10.9"     # local test; on the Pi use the Mac's hotspot IP (ipconfig getifaddr en0)
FRAME_PORT = 5555           # Pi -> laptop frames (imagezmq)
COMMAND_PORT = 5556         # laptop -> Pi commands (pyzmq)
FRAME_SIZE = (160, 120)       # grayscale size the fly eyes use (made on the laptop)
COLOR_SIZE = (320, 240)       # color frame the Pi sends (bigger, for a later vehicle check)
EYES_ZOOM = 1.25              # eyes see the centre 1/zoom of the frame: distant cars cover more facets (earlier
                              # warnings) but the view is narrower; 1.25 added ~0.1-0.2 s with no turning false alarms
CAMERA_FPS = 30             # frames per second the live camera delivers (Waymo runs pass their own 10)
CHUNK = 5                   # frames per eye-model step; smaller = less lag
WARN_THRESHOLD = 0.4        # loom level where the fly escapes; 0.4 gave earlier warnings with no extra false alarms (synthetic test)
FAKE_EYES = False           # True = frame-difference stand-in; False = flyvis fly eyes
EYES_MODEL = "flow/0000/000"  # flyvis ensemble model; needs tools/tuning/<model>.json
VEHICLE_CHECK = True        # YOLO vehicle detector on the color frames (needs ultralytics); shown, doesn't affect warnings
FAKE_BRAIN = False          # True = "loom > WARN_THRESHOLD" stand-in; False = flybrain whole-brain model
WARN_HOLD = 0.5             # seconds the warning keeps beeping after the last escape command
AUDIO_DEVICE = "plughw:CARD=G,DEV=0"  # Corsair USB headset on the Pi (aplay -l); None = system default
EAR_SIDE = 0.3              # |turn| above this beeps in one ear only; below it, both ears
SWAP_EARS = True            # camera facing backward (or at you): image left = rider's right, so swap
