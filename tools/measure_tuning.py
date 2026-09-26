"""Direction tuning of T4 (bright edges) and T5 (dark edges) in flyvis networks.

Uses flyvis's own moving-edge test (the same one tools/check_flyvis.py plots).
Takes ~12 minutes per model on a laptop CPU the first time (flyvis caches it after).
Run from the repo root:  python -m tools.measure_tuning [model ...]
Saves tools/tuning/<model>.json, which laptop/eyes.py reads.
"""
import json
import pathlib
import sys
import numpy as np
from flyvis import NetworkView
from flyvis.analysis.moving_bar_responses import (direction_selectivity_index, peak_responses,
                                                  preferred_direction)

TUNING_DIR = pathlib.Path(__file__).resolve().parent / "tuning"
CHANNELS = [(t + d, 1 if t == "T4" else 0) for t in ("T4", "T5") for d in "abcd"]   # (type, edge: 1 = ON)

def tuning_path(model):
    return TUNING_DIR / (model.replace("/", "_") + ".json")

def measure(model):
    responses = NetworkView(model).moving_edge_responses()
    dsi = direction_selectivity_index(responses)
    pd = preferred_direction(responses).squeeze("network_id")
    peak = (peak_responses(responses).set_index(sample=["angle", "width", "intensity", "speed"])
            .unstack("sample").mean(["angle", "width", "speed"]).squeeze())
    types = list(dsi.cell_type.values)
    table = {}
    for name, on in CHANNELS:
        pick = lambda x: x.isel(neuron=types.index(name)).sel(intensity=on)
        flyvis_deg = float(np.degrees(pick(pd)) % 360)
        table[name] = {
            "flyvis_deg": round(flyvis_deg, 1),
            "image_deg": round(-flyvis_deg % 360, 1),   # flyvis angles have y up, images y down
            "dsi": round(float(pick(dsi)), 3),
            "strength": round(float(pick(peak)), 4),
        }
    TUNING_DIR.mkdir(exist_ok=True)
    tuning_path(model).write_text(json.dumps(table, indent=1))
    return table

if __name__ == "__main__":
    for model in sys.argv[1:] or ["flow/0000/000"]:
        table = measure(model)
        print(model, "  weakest DSI", f"{min(c['dsi'] for c in table.values()):.2f}")
        for name, c in table.items():
            print(f"  {name}: image direction {c['image_deg']:5.0f} deg  DSI {c['dsi']:.2f}  strength {c['strength']:.3f}")
