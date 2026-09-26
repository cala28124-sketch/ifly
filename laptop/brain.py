from config import CHUNK, WARN_THRESHOLD

CAMERA_FPS = 30          # frames per second the chunks come from (sets brain time per chunk)
VOLTAGE_AT_THRESHOLD = 0.15   # LPLC2/LC4 drive per 20 ms step that reliably fires DNp01 twice in a chunk
MAX_VOLTAGE = 0.8        # most drive per step (flybrain's own cap)
ESCAPE_SPIKES = 2        # giant-fiber spikes in one chunk that count as escape (1 also happens at rest)
CHAIN = ["LC4", "LPLC2", "DNp01"]   # looming detectors -> giant fiber, reported when they spike

class FlyBrainModel:
    def __init__(self, fake=True):
        self.fake = fake
        if fake:
            return
        # real version: the MaleCNS whole-brain spiking model (166,700 neurons, ~3 ms per 20 ms step on CPU)
        from flybrain import FlyBrain
        self.brain = FlyBrain(device="auto")
        self.steps = max(1, round(CHUNK / CAMERA_FPS / self.brain.dt))   # 5 frames at 30 fps -> 8 steps
        self.loom_cells = {s: [*self.brain.cells(["LPLC2"], s), *self.brain.cells(["LC4"], s)] for s in "LR"}
        self.chain = {name: set(self.brain.cells([name]).tolist()) for name in CHAIN}
        self.chain_sides = {(name, s): set(self.brain.cells([name], s).tolist()) for name in CHAIN for s in "LR"}
        self.giant_fiber = self.chain["DNp01"]
        for _ in range(25):   # settle resting activity (0.5 s of brain time)
            self.brain.step()

    def step(self, loom_left, loom_right):
        if self.fake:
            escape = bool(max(loom_left, loom_right) > WARN_THRESHOLD)
            return {"escape": escape, "fired": ["LPLC2", "DNp01"] if escape else [], "activity": {}}

        # looming on each side drives that side's LPLC2 and LC4 neurons; a loom score at
        # WARN_THRESHOLD gives just enough drive for the giant fiber to fire reliably
        inject = []
        for side, loom in (("L", loom_left), ("R", loom_right)):
            voltage = min(loom / WARN_THRESHOLD * VOLTAGE_AT_THRESHOLD, MAX_VOLTAGE)
            if voltage > 0:
                inject.append((self.loom_cells[side], voltage))
        spiked, giant_fiber_spikes = set(), 0
        activity = {name: [0, 0] for name in CHAIN}   # spikes this chunk, [left, right], for the viewer
        for _ in range(self.steps):   # the brain keeps running between chunks
            fired = set(self.brain.step(inject=inject).tolist())
            spiked |= fired
            giant_fiber_spikes += len(fired & self.giant_fiber)   # count every spike, not just which cell
            for (name, side), cells in self.chain_sides.items():
                activity[name]["LR".index(side)] += len(fired & cells)
        escape = giant_fiber_spikes >= ESCAPE_SPIKES
        return {"escape": escape, "fired": [name for name in CHAIN if spiked & self.chain[name]] if escape else [],
                "activity": activity}
