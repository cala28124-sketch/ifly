"""Scores the fly's warnings on Waymo segments against the labeled vehicles.

Needs, per segment: the receiver's decision log (waymo/results/SEGMENT.csv, see laptop/receiver.py --log)
and Waymo's labels: waymo/lidar_box/SEGMENT.parquet (3D, preferred) or waymo/camera_box/SEGMENT.parquet.
    python -m laptop.score_waymo SEGMENT [SEGMENT ...]
    python -m laptop.score_waymo            (every segment that has a log and labels)

Ground truth: with 3D labels, each vehicle in the front camera's view that will reach the Waymo car
within TTC_MAX s is sorted by where it is now (reliable; projecting sideways drift isn't, on curves):
  collision course - in the car's path (overlapping it sideways): what the fly is built to catch, the main score
  close pass       - within PASS_MARGIN m of its side: reported separately; the fly doesn't target these yet
Motion comes from each vehicle's position relative to the car over time. Without 3D labels, Davis's
compute_ground_truth (a labeled 2D box growing fast, see GT_GROWTH) is used for everything; it also
counts parked cars the Waymo car drives past. Consecutive threat frames form one event, which ends when
the vehicle is closest.
  hit        - the fly warned between EARLY s before the event started and the event's end
  lead time  - how long before the event's end the first of those warnings came (bigger = better)
  false alarm- a run of back-to-back warnings (one beep for the rider) that is nowhere near an event
"""
import argparse
import csv
import math
import pathlib
import statistics
import pandas as pd
from laptop.waymo_player import VEHICLE_TYPE, compute_ground_truth, load_segment_boxes

# Davis's ground truth counts a vehicle as approaching when its box area grows by growth_threshold per
# frame (0.1 s). His default 1.5 only happens < 0.45 s before a vehicle would reach the camera, so almost
# nothing qualifies; 1.07 means "would reach the camera in under ~3 s" (growth = (1 + 0.1 / 3) ** 2).
GT_GROWTH = 1.07
# 3D ground truth (waymo/lidar_box): positions are metres from the Waymo car, x forward, y left
TTC_MAX = 3.0        # a vehicle that would reach the car within this many seconds...
PASS_MARGIN = 1.0    # ...and is within this many metres of its side is a close pass (in its path: collision)
EGO_FRONT = 3.0      # metres from the label origin to the Waymo car's front (approximate)
EGO_HALF_WIDTH = 1.0 # half the Waymo car's width, metres
HALF_FOV = 25        # degrees each side of straight ahead that the front camera sees
MIN_CLOSING = 0.5    # m/s: slower than this isn't approaching
EARLY = 3.0        # seconds before an event's start that a warning still counts for it
LATE = 1.0         # seconds after an event's end during which warnings aren't false alarms
MERGE_GAP = 0.5    # approaching frames less than this far apart belong to the same event
RESULTS = pathlib.Path("waymo/results")

def seconds(timestamp_micros):
    return timestamp_micros / 1e6

def lidar_ground_truth(segment):
    """{"collision": {timestamp: bool}, "close": {timestamp: bool}} from the 3D labels (see the top).
    Motion comes from each vehicle's position change relative to the car over 0.2 s, so it already
    includes the Waymo car's own movement."""
    df = pd.read_parquet(f"waymo/lidar_box/{segment}.parquet")
    col = lambda name: f"[LiDARBoxComponent].{name}"
    kinds = {"collision": set(), "close": set()}
    for _, track in df[df[col("type")] == VEHICLE_TYPE].groupby("key.laser_object_id"):
        track = track.sort_values("key.frame_timestamp_micros")
        ts = track["key.frame_timestamp_micros"].to_numpy()
        x, y = track[col("box.center.x")].to_numpy(), track[col("box.center.y")].to_numpy()
        length, width = track[col("box.size.x")].to_numpy(), track[col("box.size.y")].to_numpy()
        for i in range(2, len(ts)):
            dt = (ts[i] - ts[i - 2]) / 1e6
            if not 0 < dt <= 0.5:   # the vehicle wasn't tracked continuously
                continue
            vx, vy = (x[i] - x[i - 2]) / dt, (y[i] - y[i - 2]) / dt
            gap = x[i] - length[i] / 2 - EGO_FRONT
            in_view = x[i] > 0 and abs(math.degrees(math.atan2(y[i], x[i]))) <= HALF_FOV
            if not in_view or gap <= 0 or -vx < MIN_CLOSING:
                continue
            ttc = gap / -vx                                           # seconds until it reaches the car
            side_gap = abs(y[i]) - width[i] / 2 - EGO_HALF_WIDTH      # sideways clearance (below 0 = in its path)
            if ttc <= TTC_MAX and side_gap <= PASS_MARGIN:
                kinds["collision" if side_gap <= 0 else "close"].add(ts[i])
    frames = df["key.frame_timestamp_micros"].unique()
    return {kind: {ts: ts in found for ts in frames} for kind, found in kinds.items()}

def ground_truth(segment):
    """The 3D ground truth when the segment has 3D labels, otherwise Davis's 2D box-growth version
    (which can't tell the kinds apart, so everything counts as collision course)."""
    if pathlib.Path(f"waymo/lidar_box/{segment}.parquet").exists():
        return lidar_ground_truth(segment), "3D"
    truth = compute_ground_truth(load_segment_boxes(segment), growth_threshold=GT_GROWTH)
    return {"collision": truth, "close": {ts: False for ts in truth}}, "2D"

def approach_events(ground_truth):
    """{timestamp: approaching?} -> list of (start, end) in seconds."""
    times = sorted(seconds(ts) for ts, approaching in ground_truth.items() if approaching)
    events = []
    for t in times:
        if events and t - events[-1][1] <= MERGE_GAP:
            events[-1][1] = t
        else:
            events.append([t, t])
    return [tuple(e) for e in events]

def read_warnings(log_path):
    """The receiver's log -> (times of every decision, times of the decisions that warned), in seconds."""
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    decisions = [seconds(int(r["frame"])) for r in rows]
    warnings = [seconds(int(r["frame"])) for r in rows if r["escape"] == "1"]
    return decisions, warnings

def score(segment):
    truth, source = ground_truth(segment)
    events = approach_events(truth["collision"])
    close_passes = approach_events(truth["close"])
    decisions, warnings = read_warnings(RESULTS / f"{segment}.csv")
    step = min((b - a for a, b in zip(decisions, decisions[1:])), default=0.2)   # time between decisions

    hits, leads = 0, []
    for start, end in events:
        during = [t for t in warnings if start - EARLY <= t <= end]
        if during:
            hits += 1
            leads.append(end - during[0])
    close_warned = sum(any(start - EARLY <= t <= end for t in warnings) for start, end in close_passes)

    near_event = lambda t: any(start - EARLY <= t <= end + LATE for start, end in events + close_passes)
    false_alarms, previous = 0, None
    for t in warnings:   # back-to-back warnings are one beep for the rider
        if not near_event(t) and (previous is None or t - previous > step * 1.5):
            false_alarms += 1
        previous = t if not near_event(t) else None

    minutes = (decisions[-1] - decisions[0] + step) / 60 if decisions else 0
    return {"segment": segment, "source": source, "events": len(events), "hits": hits, "leads": leads,
            "false_alarms": false_alarms, "minutes": minutes,
            "close_passes": len(close_passes), "close_warned": close_warned}

def report(r, label=None):
    lead = f"median lead {statistics.median(r['leads']):.2f} s" if r["leads"] else "median lead -"
    rate = f"{r['false_alarms'] / r['minutes']:.1f}/min" if r["minutes"] else "-"
    print(f"{label or r['segment'][:40]:40s} {r.get('source', '  ')}  events {r['events']:2d}  hits {r['hits']:2d}  "
          f"misses {r['events'] - r['hits']:2d}  {lead:18s}  false alarms {r['false_alarms']} ({rate})  "
          f"| close passes {r['close_passes']}, warned {r['close_warned']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score the fly's warnings against Waymo's labeled vehicles.")
    parser.add_argument("segments", nargs="*", help="segment names (default: every one with a log and labels)")
    args = parser.parse_args()
    segments = args.segments or sorted(p.stem for p in RESULTS.glob("*.csv")
                                       if pathlib.Path(f"waymo/camera_box/{p.stem}.parquet").exists())
    if not segments:
        raise SystemExit("nothing to score: need waymo/results/SEGMENT.csv and waymo/camera_box/SEGMENT.parquet")
    results = [score(s) for s in segments]
    for r in results:
        report(r)
    if len(results) > 1:
        total = {"events": sum(r["events"] for r in results), "hits": sum(r["hits"] for r in results),
                 "leads": [x for r in results for x in r["leads"]],
                 "false_alarms": sum(r["false_alarms"] for r in results), "minutes": sum(r["minutes"] for r in results),
                 "close_passes": sum(r["close_passes"] for r in results), "close_warned": sum(r["close_warned"] for r in results)}
        report(total, "TOTAL")
