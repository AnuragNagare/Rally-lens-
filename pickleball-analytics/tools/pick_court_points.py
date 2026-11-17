#!/usr/bin/env python3
"""Manual court calibration, for footage where automatic line detection fails.

    python tools/pick_court_points.py --video data/input.mp4 --out court.json
    python run.py --video data/input.mp4 --court-json court.json

Click the four court corners in this order:
    1 far-left   2 far-right   3 near-right   4 near-left
Press r to restart, Enter to save, Esc to quit. Needs a desktop (uses an OpenCV window);
on a headless machine use --points instead:

    python tools/pick_court_points.py --points 745,184 1181,186 1756,876 158,871 --out court.json
"""
import argparse
import json

import cv2

ORDER = ["far-left", "far-right", "near-right", "near-left"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--points", nargs=4, help="four x,y pairs instead of clicking")
    ap.add_argument("--out", default="court.json")
    a = ap.parse_args()

    if a.points:
        pts = [[float(v) for v in p.split(",")] for p in a.points]
    else:
        cap = cv2.VideoCapture(a.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("could not read the video")
        pts = []

        def click(event, x, y, *_):
            if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
                pts.append([float(x), float(y)])

        cv2.namedWindow("court", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("court", click)
        while True:
            vis = frame.copy()
            for i, (x, y) in enumerate(pts):
                cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)
                cv2.putText(vis, f"{i + 1} {ORDER[i]}", (int(x) + 8, int(y) - 8), 0, 0.7, (0, 0, 255), 2)
            if len(pts) < 4:
                cv2.putText(vis, f"click {ORDER[len(pts)]}", (20, 40), 0, 1.0, (255, 255, 255), 2)
            cv2.imshow("court", vis)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("r"):
                pts.clear()
            elif key == 27:
                raise SystemExit("cancelled")
            elif key in (10, 13) and len(pts) == 4:
                break
    with open(a.out, "w") as f:
        json.dump({"corners_px": pts, "order": ", ".join(ORDER)}, f, indent=2)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
