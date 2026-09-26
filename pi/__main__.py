"""Start everything the Pi runs: the headset warning listener and the camera sender.

Run from the repo root:  python -m pi
Ctrl+C (or the sender stopping) stops both.
"""
import subprocess
import sys

listener = subprocess.Popen([sys.executable, "-m", "pi.commands"])
try:
    subprocess.run([sys.executable, "-m", "pi.sender"])
except KeyboardInterrupt:
    pass
finally:
    listener.terminate()
    listener.wait()
