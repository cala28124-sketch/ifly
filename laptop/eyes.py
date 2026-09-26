import json
import pathlib
import numpy as np

LOOM_SCALE = 0.3    # outward-motion strength that counts as full looming (score 1.0), set with tools.test_looming
ADAPT = 0.2         # how fast each cell's baseline follows its activity, per chunk (static scenes fade out)
RADII = (20, 40, 70)  # LPLC2-like receptive field radii in pixels (small, medium, large objects)
GATE_EPS = 0.02     # brightness change (0-1 scale) below which a column counts as unchanged
TUNING_DIR = pathlib.Path(__file__).resolve().parent.parent / "tools" / "tuning"

def load_tuning(model):
    """Each T4/T5 channel's measured direction, selectivity and strength (see tools/measure_tuning.py)."""
    path = TUNING_DIR / (model.replace("/", "_") + ".json")
    if not path.exists():
        raise FileNotFoundError(f"no tuning for {model}; run first:  python -m tools.measure_tuning {model}")
    return json.loads(path.read_text())

class FlyEyes:
    def __init__(self, fake=True, model="flow/0000/000", steps_per_frame=3, dt=1/100):
        self.fake = fake
        if fake:
            return
        # real version: pretrained flyvis optic lobe (loads in a few seconds)
        import torch
        from flyvis import NetworkView
        from flyvis.datasets.rendering import BoxEye
        self.torch = torch
        self.tuning = load_tuning(model)             # fail fast, before the slow network load
        self.net = NetworkView(model).init_network()
        self.eye = BoxEye()                          # image -> 721 hexagonal eye columns
        self.dt, self.steps = dt, steps_per_frame    # 3 steps of 10 ms ~ one 30 fps frame
        self.state = None                            # set from the first frame, then carried
        self.size = None

    def _columns(self, h, w):
        """Find each eye column's T4/T5 cells and where it sits in the image; build LPLC2-like units."""
        nodes = self.net.connectome.nodes
        types = np.array([t.decode() if isinstance(t, bytes) else str(t) for t in nodes["type"][:]])
        uv = list(zip(nodes["u"][:], nodes["v"][:]))
        # the stimulus feeds hexal j into R1 cell input_index[0][j]; that fixes hexal j's (u, v)
        hex_uv = [uv[i] for i in np.asarray(self.net.stimulus.input_index)[0]]
        self.cells = {}   # T4/T5 cells of each subtype, in hexal order
        for d in "abcd":
            t4 = {uv[i]: i for i in np.where(types == f"T4{d}")[0]}
            t5 = {uv[i]: i for i in np.where(types == f"T5{d}")[0]}
            self.cells[d] = (np.array([t4[p] for p in hex_uv]), np.array([t5[p] for p in hex_uv]))
        # each hexal's pixel position: show the eye horizontal and vertical ramps
        ramps = np.stack([np.tile(np.arange(w, dtype=np.float32), (h, 1)),
                          np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))])[:, None]
        x, y = self.x, self.y = self.eye(self.torch.tensor(ramps))[:, 0, 0].numpy()
        # LPLC2-like units: at each centre and radius, four quadrants (left, right, up, down);
        # each quadrant averages the motion component pointing away from the centre
        units, arms = [], []
        for r in RADII:
            for cy in np.arange(r / 2, h, r / 2):
                for cx in np.arange(r / 2, w, r / 2):
                    dx, dy = x - cx, y - cy
                    dist = np.hypot(dx, dy) + 1e-6
                    near = dist < r
                    quads = [near & (dx < -abs(dy)), near & (dx > abs(dy)),
                             near & (dy < -abs(dx)), near & (dy > abs(dx))]
                    if min(q.sum() for q in quads) < 3:
                        continue
                    units.append(cx < w / 2)
                    arms.append([np.stack([q * dx / dist, q * dy / dist]) / q.sum() for q in quads])
        self.unit_left = np.array(units)
        self.arms = np.array(arms)   # (units, 4 quadrants, 2 [x, y], 721) outward-projection weights
        self.size = (h, w)

    def step(self, frames):
        """frames: list of 160x120 grayscale arrays. Returns the eyes contract."""
        if self.fake:
            diff = np.abs(frames[-1].astype(int) - frames[0].astype(int))
            half = diff.shape[1] // 2
            return {"t4": [0, 0, 0, 0],
                    "loom_left": float(min(diff[:, :half].mean() / 40, 1.0)),
                    "loom_right": float(min(diff[:, half:].mean() / 40, 1.0))}

        if self.size != frames[0].shape:
            self._columns(*frames[0].shape)
        movie = self.torch.tensor(np.array(frames, np.float32)[None] / 255.0)   # (1, frames, h, w)
        stim = self.eye(movie).repeat_interleave(self.steps, dim=1)             # (1, steps, 1, 721)
        with self.torch.no_grad():
            if self.state is None:   # settle on the first real image so it doesn't look like a flash
                self.state = self.net.fade_in_state(1.0, self.dt, stim[:, 0])
                self.base = self.state.nodes.activity[0].numpy().copy()   # copy: numpy() shares torch memory
                self.last_light = stim[0, :1, 0].numpy()
            states = self.net.simulate(stim, self.dt, initial_state=self.state, as_states=True)
        self.state = states[-1]
        activity = np.stack([s.nodes.activity[0].numpy() for s in states])     # (steps, cells)
        response = self.response = np.maximum(activity - self.base, 0).mean(0)  # above the adapted baseline
        self.base += ADAPT * (activity.mean(0) - self.base)

        # which edge type actually passed each eye column: brighter -> trust T4, darker -> trust T5
        # (each family also fires backwards at the other edge type; this gate filters that out)
        light = np.concatenate([self.last_light, stim[0, :, 0].numpy()])       # (steps + 1, 721)
        change = np.diff(light, axis=0)
        brighter, darker = np.maximum(change, 0).sum(0), np.maximum(-change, 0).sum(0)
        gate = {"T4": brighter / (brighter + darker + GATE_EPS), "T5": darker / (brighter + darker + GATE_EPS)}
        self.last_light = light[-1:]

        # motion arrow per eye column: each channel scaled by its strength, minus what its group
        # (T4 = bright edges, T5 = dark edges) shares, then pointed in its measured direction and
        # weighted by how direction-selective it is (a channel with no preference adds nothing)
        flow = np.zeros((2, len(self.x)))
        for kind in ("T4", "T5"):
            tune = [self.tuning[kind + d] for d in "abcd"]
            m = np.array([response[self.cells[d][kind == "T5"]] / t["strength"] for d, t in zip("abcd", tune)])
            m = (m - m.mean(0)) * gate[kind]
            angle = np.radians([t["image_deg"] for t in tune])
            weight = np.array([t["dsi"] for t in tune])
            flow += np.stack([(weight * np.cos(angle)) @ m, (weight * np.sin(angle)) @ m])
        # like LPLC2: outward motion in each quadrant around a centre; a unit needs all four
        self.arm_out = np.einsum("uqkh,kh->qu", self.arms, flow)               # (4 quadrants, units)
        outward = self.arm_out.min(0)
        loom = lambda side: float(np.clip(outward[side].max() / LOOM_SCALE, 0, 1))
        return {"t4": [float(response[self.cells[d][0]].mean()) for d in "abcd"],
                "loom_left": loom(self.unit_left),
                "loom_right": loom(~self.unit_left)}
