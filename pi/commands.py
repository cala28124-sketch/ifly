"""Runs on the Pi: listens for the laptop's warnings and beeps through the USB headset,
in the ear on the side the threat is on (both ears when it's in the middle).

Run from the repo root:  python -m pi.commands          (listen to the laptop)
                         python -m pi.commands --test   (beep left, right, then both, to check the headset)
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
from config import LAPTOP_IP, COMMAND_PORT, WARN_HOLD, AUDIO_DEVICE, EAR_SIDE, SWAP_EARS

BEEP_HZ, BEEP_S, GAP_S = 880, 0.2, 0.1   # pitch, beep length, silence between beeps

def make_beep(ear):
    """Write a short stereo sine-wave beep for one ear ("left", "right") or "both"; return its path."""
    rate = 44100
    n = int(rate * BEEP_S)
    fade = int(rate * 0.005)   # 5 ms fade in/out so it doesn't click
    tone = [int(12000 * min(1, i / fade, (n - i) / fade) * math.sin(2 * math.pi * BEEP_HZ * i / rate))
            for i in range(n)]
    left, right = ear in ("left", "both"), ear in ("right", "both")
    samples = array.array("h", (s for t in tone for s in (t * left, t * right)))   # interleaved L, R
    path = pathlib.Path(tempfile.gettempdir()) / f"fly_beep_{ear}.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return path

EARS = ("left", "right", "both")
beeps = {ear: make_beep(ear) for ear in EARS}
if shutil.which("aplay"):
    play = {ear: ["aplay", "-q"] + (["-D", AUDIO_DEVICE] if AUDIO_DEVICE else []) + [str(beeps[ear])] for ear in EARS}
elif shutil.which("afplay"):
    play = {ear: ["afplay", str(beeps[ear])] for ear in EARS}
else:
    play = None
    print("no audio player found; printing instead")

alarm = threading.Event()   # set = beeping
ear = "both"                # which ear the beeps go to right now

def ear_for(turn):
    """turn: -1 = threat on the image's left ... +1 = right (from the laptop)."""
    if abs(turn) <= EAR_SIDE:
        return "both"
    image_side = "left" if turn < 0 else "right"
    if SWAP_EARS:   # facing backward, the image's left is the rider's right
        return {"left": "right", "right": "left"}[image_side]
    return image_side

def beeper():
    """Background thread: beep over and over while the alarm is set."""
    reported = False
    while True:
        alarm.wait()
        if play:
            result = subprocess.run(play[ear], stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                if not reported:   # say it once, not on every beep
                    print(f"audio failed: {result.stderr.strip()}\n"
                          "  check the headset with  aplay -l  and set AUDIO_DEVICE in config.py")
                    reported = True
                time.sleep(BEEP_S)   # failed plays return instantly; keep the beep rhythm
        else:
            print(f"BEEP ({ear})")
        time.sleep(GAP_S)

threading.Thread(target=beeper, daemon=True).start()

def warning(on):
    print(f"WARN ON ({ear} ear)" if on else "warn off")
    alarm.set() if on else alarm.clear()

if "--test" in sys.argv:
    for ear in EARS:   # the beeper thread reads the global ear
        warning(True)
        time.sleep(1.0)
        warning(False)
        time.sleep(0.5)
    sys.exit()

sub = zmq.Context().socket(zmq.SUB)
sub.connect(f"tcp://{LAPTOP_IP}:{COMMAND_PORT}")
sub.setsockopt(zmq.SUBSCRIBE, b"")   # receive every message
print(f"listening for warnings from {LAPTOP_IP}:{COMMAND_PORT}")

last_warn, is_on = 0.0, False
try:
    while True:
        if sub.poll(100):   # wait up to 100 ms for a command
            command = sub.recv_json()
            if command.get("escape"):
                last_warn = time.time()
                if ear != ear_for(command.get("turn", 0.0)):
                    ear = ear_for(command.get("turn", 0.0))
                    if is_on:
                        print(f"  now {ear} ear")
        # stay on for WARN_HOLD after the last warning, so it doesn't stutter between chunks;
        # this also stops it if the laptop stops sending
        should_be_on = time.time() - last_warn < WARN_HOLD
        if should_be_on != is_on:
            warning(should_be_on)
            is_on = should_be_on
except KeyboardInterrupt:
    warning(False)
