#!/usr/bin/env python3
"""Score detected shots against hand labels.

    python tools/evaluate.py --shots output/shots.csv --labels data/labels_input.json

A detected shot is correct if a labelled hit by the same player lies within --tolerance
frames. Prints precision, recall and the timing error. Label your own clip the same way
(a JSON list of [frame, "near"|"far"]) to get an accuracy number you can quote.
"""
import argparse
import csv
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", default="output/shots.csv")
    ap.add_argument("--labels", default="data/labels_input.json")
    ap.add_argument("--tolerance", type=int, default=8, help="frames")
    a = ap.parse_args()

    with open(a.shots) as f:
        pred = [(int(r["frame"]), r["player"]) for r in csv.DictReader(f)]
    truth = [(int(fr), side) for fr, side in json.load(open(a.labels))["hits"]]

    used, errors, wrong = set(), [], []
    for pf, ps in pred:
        best = None
        for k, (tf, ts) in enumerate(truth):
            if k in used or ts != ps or abs(tf - pf) > a.tolerance:
                continue
            if best is None or abs(tf - pf) < abs(truth[best][0] - pf):
                best = k
        if best is None:
            wrong.append((pf, ps))
        else:
            used.add(best)
            errors.append(abs(truth[best][0] - pf))
    missed = [t for k, t in enumerate(truth) if k not in used]

    tp = len(used)
    print(f"labelled hits   : {len(truth)}")
    print(f"detected shots  : {len(pred)}")
    print(f"correct         : {tp}")
    print(f"precision       : {tp / max(len(pred), 1):.0%}")
    print(f"recall          : {tp / max(len(truth), 1):.0%}")
    if errors:
        print(f"timing error    : mean {sum(errors) / len(errors):.1f} frames, worst {max(errors)} frames")
    if wrong:
        print(f"false detections: {wrong}")
    if missed:
        print(f"missed hits     : {missed}")


if __name__ == "__main__":
    main()
