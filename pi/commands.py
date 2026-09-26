"""Runs on the Pi: listens for the laptop's warnings and beeps through the USB headset.

Run from the repo root:  python -m pi.commands          (listen to the laptop)
                         python -m pi.commands --test   (beep for ~1.5 s to check the headset)
Uses aplay (built into Raspberry Pi OS); on a Mac it uses afplay, elsewhere it prints instead.
"""
import array
import math
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
import zmq
from config import LAPTOP_IP, COMMAND_PORT, WARN_HOLD, AUDIO_DEVICE

BEEP_HZ, BEEP_S, GAP_S = 880, 0.2, 0.1   # pitch, beep length, silence between beeps

def make_beep():
    """Write a short sine-wave beep to a .wav file and return its path."""
    rate = 44100
    n = int(rate * BEEP_S)
    fade = int(rate * 0.005)   # 5 ms fade in/out so it doesn't click
    samples = array.array("h", (int(12000 * min(1, i / fade, (n - i) / fade)
                                    * math.sin(2 * math.pi * BEEP_HZ * i / rate)) for i in range(n)))
    path = pathlib.Path(tempfile.gettempdir()) / "fly_beep.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return path

beep = make_beep()
if shutil.which("aplay"):
    play = ["aplay", "-q"] + (["-D", AUDIO_DEVICE] if AUDIO_DEVICE else []) + [str(beep)]
elif shutil.which("afplay"):
    play = ["afplay", str(beep)]
else:
    play = None
    print("no audio player found; printing instead")

alarm = threading.Event()   # set = beeping

def beeper():
    """Background thread: beep over and over while the alarm is set."""
    while True:
        alarm.wait()
        if play:
            subprocess.run(play)
        else:
            print("BEEP")
        time.sleep(GAP_S)

threading.Thread(target=beeper, daemon=True).start()

def warning(on):
    print("WARN ON" if on else "warn off")
    alarm.set() if on else alarm.clear()

if "--test" in sys.argv:
    warning(True)
    time.sleep(1.5)
    warning(False)
    time.sleep(0.5)   # let the last beep finish
    sys.exit()

sub = zmq.Context().socket(zmq.SUB)
sub.connect(f"tcp://{LAPTOP_IP}:{COMMAND_PORT}")
sub.setsockopt(zmq.SUBSCRIBE, b"")   # receive every message
print(f"listening for warnings from {LAPTOP_IP}:{COMMAND_PORT}")

last_warn, is_on = 0.0, False
try:
    while True:
        if sub.poll(100):   # wait up to 100 ms for a command
            if sub.recv_json().get("escape"):
                last_warn = time.time()
        # stay on for WARN_HOLD after the last warning, so it doesn't stutter between chunks;
        # this also stops it if the laptop stops sending
        should_be_on = time.time() - last_warn < WARN_HOLD
        if should_be_on != is_on:
            warning(should_be_on)
            is_on = should_be_on
except KeyboardInterrupt:
    warning(False)
