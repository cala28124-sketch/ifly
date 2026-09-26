import numpy as np

class FlyEyes:
    def __init__(self, fake=True):
        self.fake = fake
        # real version: load BoxEye + pretrained flyvis network here (Tech reference)

    def step(self, frames):
        """frames: list of 160x120 grayscale arrays. Returns the eyes contract."""
        if self.fake:
            diff = np.abs(frames[-1].astype(int) - frames[0].astype(int))
            half = diff.shape[1] // 2
            return {"t4": [0, 0, 0, 0],
                    "loom_left": float(min(diff[:, :half].mean() / 40, 1.0)),
                    "loom_right": float(min(diff[:, half:].mean() / 40, 1.0))}
        # real version: flyvis on the chunk, carry state, compute looming from T4/T5
