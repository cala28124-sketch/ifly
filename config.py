LAPTOP_IP = "172.20.10.9"     # local test; on the Pi use the Mac's hotspot IP (ipconfig getifaddr en0)
FRAME_PORT = 5555           # Pi -> laptop frames (imagezmq)
COMMAND_PORT = 5556         # laptop -> Pi commands (pyzmq)
FRAME_SIZE = (160, 120)
CHUNK = 5                   # frames per eye-model step; smaller = less lag
WARN_THRESHOLD = 0.6        # tune on webcam and Waymo clips
