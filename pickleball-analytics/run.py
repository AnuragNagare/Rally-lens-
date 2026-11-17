#!/usr/bin/env python3
"""Turn one fixed-camera pickleball video into an analytics dashboard video + stats.

    python run.py --video data/input.mp4 --out output
"""
import argparse
import json

from pickle_analytics.pipeline import run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default="data/input.mp4", help="input video (fixed camera behind a baseline)")
    ap.add_argument("--out", default="output", help="output folder")
    ap.add_argument("--court-json", help="manual court calibration from tools/pick_court_points.py")
    ap.add_argument("--pose-weights", default="models/yolo11s-pose.pt",
                    help="Ultralytics pose weights (downloaded automatically if missing)")
    ap.add_argument("--ball-weights", help="optional custom YOLO ball detector; default is the classical detector")
    ap.add_argument("--imgsz", type=int, default=1280, help="inference size (higher = better on small players)")
    ap.add_argument("--start", type=float, help="start time in seconds")
    ap.add_argument("--end", type=float, help="end time in seconds")
    ap.add_argument("--near-name", default="Near player")
    ap.add_argument("--far-name", default="Far player")
    ap.add_argument("--title", default="PICKLEBALL AI ANALYTICS")
    ap.add_argument("--no-cache", action="store_true", help="re-run pose estimation even if cached")
    a = ap.parse_args()
    summary = run(a.video, a.out, a.court_json, a.pose_weights, a.ball_weights, a.imgsz, a.start, a.end,
                  a.near_name, a.far_name, a.title, use_cache=not a.no_cache)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
