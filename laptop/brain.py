from config import WARN_THRESHOLD

class FlyBrainModel:
    def __init__(self, fake=True):
        self.fake = fake
        # real version: flybrain.FlyBrain(), look up LC4/LPLC2 and DNp01 (Tech reference)

    def step(self, loom_left, loom_right):
        if self.fake:
            escape = bool(max(loom_left, loom_right) > WARN_THRESHOLD)
            return {"escape": escape, "fired": ["LPLC2", "DNp01"] if escape else []}
        # real version: inject looming into LC4/LPLC2, step, check if DNp01 fired
