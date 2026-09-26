"""Runs on the Pi: listens for the laptop's warnings and drives the LED and buzzer.

Run from the repo root:  python -m pi.commands          (listen to the laptop)
                         python -m pi.commands --test   (blink/beep 3 times to check the wiring)
On a machine without GPIO (e.g. a Mac) it prints ON/OFF instead, so it can be tried anywhere.
"""
import sys
import time
import zmq
from config import LAPTOP_IP, COMMAND_PORT, LED_PIN, BUZZER_PIN, WARN_HOLD

try:
    from gpiozero import LED, Buzzer
    led, buzzer = LED(LED_PIN), Buzzer(BUZZER_PIN)
except Exception as e:   # no GPIO here: print instead of switching pins
    print(f"no GPIO ({e.__class__.__name__}); printing instead")
    led = buzzer = None

def warning(on):
    if led is None:
        print("WARN ON" if on else "warn off")
        return
    for part in (led, buzzer):
        part.on() if on else part.off()

if "--test" in sys.argv:
    for _ in range(3):
        warning(True); time.sleep(0.3)
        warning(False); time.sleep(0.3)
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
        # stay on for WARN_HOLD after the last warning, so it doesn't flicker between chunks;
        # this also switches it off if the laptop stops sending
        should_be_on = time.time() - last_warn < WARN_HOLD
        if should_be_on != is_on:
            warning(should_be_on)
            is_on = should_be_on
except KeyboardInterrupt:
    warning(False)
