LAPTOP_IP = "172.20.10.9"     # local test; on the Pi use the Mac's hotspot IP (ipconfig getifaddr en0)
FRAME_PORT = 5555           # Pi -> laptop frames (imagezmq)
COMMAND_PORT = 5556         # laptop -> Pi commands (pyzmq)
FRAME_SIZE = (160, 120)       # grayscale size the fly eyes use (made on the laptop)
COLOR_SIZE = (320, 240)       # color frame the Pi sends (bigger, for a later vehicle check)
CHUNK = 5                   # frames per eye-model step; smaller = less lag
WARN_THRESHOLD = 0.6        # tune on webcam and Waymo clips
FAKE_EYES = False           # True = frame-difference stand-in; False = flyvis fly eyes
EYES_MODEL = "flow/0000/000"  # flyvis ensemble model; needs tools/tuning/<model>.json
FAKE_BRAIN = False          # True = "loom > WARN_THRESHOLD" stand-in; False = flybrain whole-brain model
WARN_HOLD = 0.5             # seconds the warning keeps beeping after the last escape command
AUDIO_DEVICE = "plughw:CARD=G,DEV=0"  # Corsair USB headset on the Pi (aplay -l); None = system default
