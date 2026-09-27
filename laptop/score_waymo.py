"""Scores the fly's warnings on Waymo segments against the labeled vehicles.

Needs, per segment: the receiver's decision log (waymo/results/SEGMENT.csv, see laptop/receiver.py --log)
and Waymo's labels (waymo/camera_box/SEGMENT.parquet).
    python -m laptop.score_waymo SEGMENT [SEGMENT ...]
    python -m laptop.score_waymo            (every segment that has a log and labels)

Approach events come from Davis's compute_ground_truth (a labeled vehicle's box growing fast, see GT_GROWTH):
consecutive approaching frames form one event, which ends when the vehicle is closest.
  hit        - the fly warned between EARLY s before the event started and the event's end
  lead time  - how long before the event's end the first of those warnings came (bigger = better)
  false alarm- a run of back-to-back warnings (one beep for the rider) that is nowhere near an event
"""
import argparse
import csv
import pathlib
import statistics
from laptop.waymo_player import compute_ground_truth, load_segment_boxes

# Davis's ground truth counts a vehicle as approaching when its box area grows by growth_threshold per
# frame (0.1 s). His default 1.5 only happens < 0.45 s before a vehicle would reach the camera, so almost
# nothing qualifies; 1.07 means "would reach the camera in under ~3 s" (growth = (1 + 0.1 / 3) ** 2).
GT_GROWTH = 1.07
EARLY = 3.0        # seconds before an event's start that a warning still counts for it
LATE = 1.0         # seconds after an event's end during which warnings aren't false alarms
MERGE_GAP = 0.5    # approaching frames less than this far apart belong to the same event
RESULTS = pathlib.Path("waymo/results")

def seconds(timestamp_micros):
    return timestamp_micros / 1e6

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
    events = approach_events(compute_ground_truth(load_segment_boxes(segment), growth_threshold=GT_GROWTH))
    decisions, warnings = read_warnings(RESULTS / f"{segment}.csv")
    step = min((b - a for a, b in zip(decisions, decisions[1:])), default=0.2)   # time between decisions

    hits, leads = 0, []
    for start, end in events:
        during = [t for t in warnings if start - EARLY <= t <= end]
        if during:
            hits += 1
            leads.append(end - during[0])

    near_event = lambda t: any(start - EARLY <= t <= end + LATE for start, end in events)
    false_alarms, previous = 0, None
    for t in warnings:   # back-to-back warnings are one beep for the rider
        if not near_event(t) and (previous is None or t - previous > step * 1.5):
            false_alarms += 1
        previous = t if not near_event(t) else None

    minutes = (decisions[-1] - decisions[0] + step) / 60 if decisions else 0
    return {"segment": segment, "events": len(events), "hits": hits, "leads": leads,
            "false_alarms": false_alarms, "minutes": minutes}

def report(r, label=None):
    lead = f"median lead {statistics.median(r['leads']):.2f} s" if r["leads"] else "median lead -"
    rate = f"{r['false_alarms'] / r['minutes']:.1f}/min" if r["minutes"] else "-"
    print(f"{label or r['segment'][:40]:40s}  events {r['events']:2d}  hits {r['hits']:2d}  "
          f"misses {r['events'] - r['hits']:2d}  {lead:18s}  false alarms {r['false_alarms']} ({rate})")

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
                 "false_alarms": sum(r["false_alarms"] for r in results), "minutes": sum(r["minutes"] for r in results)}
        report(total, "TOTAL")
