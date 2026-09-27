"""Whole-brain view: a front view of the fly brain where neurons firing more than usual light up.

About 20% of the brain spikes in every chunk just from background activity, so drawing every spike
would hide the signal. Instead each neuron compares its recent firing with its long-run normal, and
only neurons firing well above normal for several chunks in a row glow (orange). The escape circuit is always highlighted:
LPLC2/LC4 in yellow when they spike, and the two giant fibers (DNp01) as orange dots when they spike,
red when they spike enough to escape.
Positions come from flybrain's data (about 140,000 of the 166,700 neurons have one).
"""
import pathlib
import cv2
import numpy as np
from laptop.brain import ESCAPE_SPIKES

WIDTH = 700                  # window width in pixels (height follows the brain's shape)
RECENT = 0.5                 # how fast each neuron's recent firing follows its activity, per chunk
NORMAL = 0.05                # how fast its long-run "normal" firing follows (much slower)
GLOW = 0.6                   # recent firing this far above normal counts as unusual (needs ~3 chunks in a row)

class BrainMap:
    def __init__(self):
        from flybrain.data import DATA
        meta = np.load(pathlib.Path(DATA) / "brain.npz")
        pos = meta["positions"]
        self.has_pos = ~np.isnan(pos).any(1)
        x, y = pos[self.has_pos, 0], pos[self.has_pos, 1]
        x = x.max() - x    # mirror so the fly's left is on the screen's left, like the circuit panel
        scale = (WIDTH - 20) / (x.max() - x.min())
        self.height = int((y.max() - y.min()) * scale) + 60
        self.px = ((x - x.min()) * scale + 10).astype(int)
        self.py = ((y - y.min()) * scale + 50).astype(int)
        # neuron index in the brain -> position in the arrays above (-1 = no position)
        self.slot = np.full(len(pos), -1)
        self.slot[self.has_pos] = np.arange(self.has_pos.sum())
        cell_type = meta["cell_type"][self.has_pos]
        self.looming = np.isin(cell_type, ["LPLC2", "LC4"])
        self.giant_fiber = np.flatnonzero(cell_type == "DNp01")
        self.giant_fiber_side = meta["side"][self.has_pos][self.giant_fiber]   # "L" / "R"
        self.recent = self.normal = None   # set from the first chunk
        self.base = np.zeros((self.height, WIDTH, 3), np.uint8)
        self.base[self.py, self.px] = (70, 70, 70)   # every positioned neuron, dim gray
        for i in self.giant_fiber:                   # giant fiber outlines, always visible
            cv2.circle(self.base, (self.px[i], self.py[i]), 7, (0, 0, 160), 1)
        cv2.putText(self.base, "fly brain (front view)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.base, "LEFT", (10, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(self.base, "RIGHT", (WIDTH - 60, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def draw(self, spiked, giant_fiber_spikes=(0, 0)):
        """spiked: indices of every neuron that fired this chunk; giant_fiber_spikes: DNp01 spike counts
        this chunk, [left, right]. Returns the image."""
        slots = self.slot[np.asarray(spiked, int)]
        now = np.zeros(len(self.px), np.float32)
        now[slots[slots >= 0]] = 1
        if self.normal is None:
            self.recent, self.normal = now.copy(), now.copy()
        self.recent += RECENT * (now - self.recent)
        self.normal += NORMAL * (now - self.normal)
        unusual = self.recent - self.normal > GLOW
        img = self.base.copy()
        for dy in (0, 1):   # 2x2 dots so they're visible
            for dx in (0, 1):
                img[self.py[unusual] + dy, self.px[unusual] + dx] = (0, 140, 255)   # orange: firing above normal
        hot = (now > 0) & self.looming
        for i in np.flatnonzero(hot):
            cv2.circle(img, (self.px[i], self.py[i]), 2, (0, 255, 255), -1)   # yellow: LPLC2 / LC4 spiking
        for i, side in zip(self.giant_fiber, self.giant_fiber_side):
            if now[i]:   # orange: giant fiber spiked; red: spiked enough to escape
                count = giant_fiber_spikes["LR".index(side)] if side in ("L", "R") else 1
                color = (0, 0, 255) if count >= ESCAPE_SPIKES else (0, 165, 255)
                cv2.circle(img, (self.px[i], self.py[i]), 7, color, -1)
        return img
