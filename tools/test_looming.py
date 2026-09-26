"""Would the real eyes + fake brain warn at the right times?

Plays synthetic clips (still, sliding sideways, growing) through laptop/eyes.py and counts,
per clip, how many 5-frame chunks score above WARN_THRESHOLD on each side.
Run from the repo root:  python -m tools.test_looming [model]
Passes when still/sideways clips never warn, approaches warn in most chunks while growing,
and an approach on the left warns on the left only.
"""
import sys
import numpy as np
from config import WARN_THRESHOLD
from laptop.eyes import FlyEyes

H, W = 120, 160

def clip(kind, dark, n=40):
    """A square on a plain background: still, sliding right, or growing (approaching)."""
    frames = []
    for i in range(n):
        f = np.full((H, W), 200 if dark else 40, np.uint8)
        if kind == "still":
            cx, r = 80, 15
        elif kind == "sideways":
            cx, r = 10 + 3 * i, 15
        elif kind == "approach":
            cx, r = 80, 3 + int(1.6 * i)
        elif kind == "approach_left":
            cx, r = 40, 3 + int(1.0 * i)
        f[max(0, 60 - r):60 + r, max(0, cx - r):cx + r] = 20 if dark else 230
        frames.append(f)
    return frames

CLIPS = [("still", False), ("sideways", False), ("sideways", True),
         ("approach", False), ("approach", True), ("approach_left", False)]

def run(model):
    eyes = FlyEyes(fake=False, model=model)
    results = {}
    for kind, dark in CLIPS:
        eyes.state = None
        frames = clip(kind, dark)
        scores = []
        for k in range(0, len(frames), 5):
            seen = eyes.step(frames[k:k + 5])
            scores.append((seen["loom_left"], seen["loom_right"]))
        results[kind + (" dark" if dark else "")] = np.array(scores[1:])   # first chunk = settling
    return results

def verdict(results):
    warn = {c: s >= WARN_THRESHOLD for c, s in results.items()}
    growing = slice(0, 5)   # chunks while the square is still growing inside the image
    checks = {
        "still never warns": not warn["still"].any(),
        "sideways never warns": not warn["sideways"].any() and not warn["sideways dark"].any(),
        "approach warns (most chunks)": all(warn[c][growing].any(1).mean() >= 0.6
                                            for c in ("approach", "approach dark")),
        "left approach: left only": warn["approach_left"][:, 0].mean() >= 0.6
                                    and not warn["approach_left"][:, 1].any(),
    }
    return checks

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "flow/0000/000"
    results = run(model)
    for c, s in results.items():
        print(f"{c:15s} " + "  ".join(f"{l:.2f}/{r:.2f}" for l, r in s))
    checks = verdict(results)
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print("ALL PASS" if all(checks.values()) else "SOME FAILED")
