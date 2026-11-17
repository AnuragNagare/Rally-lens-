"""End-to-end pipeline: video in -> annotated dashboard video + stats files out."""
from __future__ import annotations

import csv
import json
import pickle
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from .ball import (detect_candidates_classical, detect_candidates_yolo, play_region_mask,
                   suppress_body_candidates, track_ball)
from .court import Court, background_frame, detect_court_corners
from .players import detect_people, far_half_roi, track_players
from .render import H, W, DashboardRenderer
from .shots import detect_shots


def _log(msg):
    print(f"[pickle-analytics] {msg}", flush=True)


def _trim(video: Path, out: Path, start: float | None, end: float | None) -> Path:
    """Cut [start, end] seconds into a new file (re-encoded so frame indices are exact)."""
    if start is None and end is None:
        return video
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(video)]
    if end is not None:
        cmd += ["-t", str(end - (start or 0))]
    cmd += ["-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True)
    return out


class _Writer:
    """H.264 via ffmpeg when available (plays everywhere), else OpenCV's mp4v."""

    def __init__(self, path: Path, fps: float):
        self.proc = None
        if shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                 "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    def write(self, frame):
        if self.proc:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.cv.write(frame)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()


def run(video: str, out_dir: str = "output", court_json: str | None = None,
        pose_weights: str = "yolo11s-pose.pt", ball_weights: str | None = None,
        imgsz: int = 1280, start: float | None = None, end: float | None = None,
        near_name: str = "Near player", far_name: str = "Far player",
        title: str = "PICKLEBALL AI ANALYTICS", use_cache: bool = True) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = _trim(Path(video), out / "_clip.mp4", start, end)

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # 1. court ------------------------------------------------------------------------
    _log("1/6 court: building background plate and locating the lines")
    bg = background_frame(str(src))
    if court_json:
        court = Court.load(court_json)
    else:
        court = Court(detect_court_corners(bg))
    fit = court.line_fit_score(bg)
    court.save(out / "court.json")
    _log(f"    court line fit {fit:.0%} (share of the projected lines that land on paint)")
    if fit < 0.8:
        _log("    WARNING: poor fit. Run tools/pick_court_points.py and pass --court-json.")

    # 2. players ----------------------------------------------------------------------
    cache = out / "_pose_cache.pkl"
    key = (str(src), Path(src).stat().st_size, pose_weights, imgsz)
    detections = None
    if use_cache and cache.exists():
        saved = pickle.loads(cache.read_bytes())
        if saved.get("key") == key:
            detections = saved["detections"]
            _log("2/6 players: using cached pose detections")
    if detections is None:
        _log("2/6 players: running pose estimation (the slow step on CPU)")
        detections = detect_people(str(src), pose_weights, imgsz=imgsz, far_roi=far_half_roi(court, bg.shape),
                                   progress=lambda i: _log(f"    frame {i}"))
        cache.write_bytes(pickle.dumps({"key": key, "detections": detections}))
    players = track_players(detections, court, sh, fps)

    # 3. ball -------------------------------------------------------------------------
    if ball_weights:
        _log("3/6 ball: YOLO detector")
        cands = detect_candidates_yolo(str(src), ball_weights, imgsz=imgsz)
    else:
        _log("3/6 ball: classical detector (colour + background difference)")
        cands = detect_candidates_classical(str(src), bg, play_region_mask(court, bg.shape))
        cands = suppress_body_candidates(cands, detections)
    ball = track_ball(cands, fps, sh, person_boxes=[d["boxes"] for d in detections])

    # 4. shots ------------------------------------------------------------------------
    _log("4/6 shots: finding hits and per-shot speed")
    shots, rallies = detect_shots(ball, players, fps)

    # 5. stats files ------------------------------------------------------------------
    _log("5/6 writing stats")
    n = len(ball.xy)
    speeds = [s.speed_kmh for s in shots if s.speed_kmh is not None]
    summary = {
        "video": str(video), "fps": fps, "frames": n, "duration_s": round(n / fps, 2),
        "court_line_fit": round(fit, 3),
        "rallies": [{"rally": i + 1, "start_s": round(a / fps, 2), "end_s": round(b / fps, 2),
                     "shots": sum(1 for s in shots if s.rally == i + 1)} for i, (a, b) in enumerate(rallies)],
        "shots_total": len(shots),
        "avg_shot_speed_kmh": round(float(np.mean(speeds)), 1) if speeds else None,
        "fastest_shot_kmh": max(speeds) if speeds else None,
        "ball": {"frames_detected": int((ball.state == 1).sum()),
                 "frames_interpolated": int((ball.state == 2).sum()),
                 "frames_missing": int((ball.state == 0).sum())},
        "players": {},
        "notes": "Shot speed is the average over the flight (hitter position to receiver position, "
                 "divided by time between hits). It is an estimate, lower than a radar reading.",
    }
    for side, tr in players.items():
        y = tr.court[:, 1]
        seen = ~np.isnan(y)
        summary["players"][side] = {
            "name": near_name if side == "near" else far_name,
            "frames_tracked": int(tr.seen.sum()),
            "distance_m": round(float(tr.dist[-1]), 1),
            "top_speed_kmh": round(float(tr.speed.max()), 1),
            "avg_speed_kmh": round(float(tr.speed[seen].mean()), 1) if seen.any() else None,
            "shots": sum(1 for s in shots if s.side == side),
            "time_at_kitchen_line_s": round(float(((np.abs(y - 6.705) <= 3.03) & seen).sum() / fps), 1),
        }
    (out / "stats.json").write_text(json.dumps(summary, indent=2))
    with open(out / "shots.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["shot", "rally", "shot_in_rally", "frame", "time_s", "player", "zone", "hitter_x_m",
                     "hitter_y_m", "receiver_x_m", "receiver_y_m", "distance_m", "avg_speed_kmh"])
        for s in shots:
            rx, ry = s.receiver_xy if s.receiver_xy else ("", "")
            wr.writerow([s.index, s.rally, s.number_in_rally, s.frame, s.time_s, s.side, s.zone,
                         *s.hitter_xy, rx, ry, s.distance_m or "", s.speed_kmh or ""])
    with open(out / "tracks.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "time_s", "near_x_m", "near_y_m", "near_speed_kmh", "far_x_m", "far_y_m",
                     "far_speed_kmh", "ball_x_px", "ball_y_px", "ball_state"])
        fmt = lambda v: "" if np.isnan(v) else round(float(v), 2)
        for i in range(n):
            wr.writerow([i, round(i / fps, 3),
                         fmt(players["near"].court[i, 0]), fmt(players["near"].court[i, 1]), fmt(players["near"].speed[i]),
                         fmt(players["far"].court[i, 0]), fmt(players["far"].court[i, 1]), fmt(players["far"].speed[i]),
                         fmt(ball.xy[i, 0]), fmt(ball.xy[i, 1]), ("missing", "detected", "interpolated")[ball.state[i]]])

    # 6. render -----------------------------------------------------------------------
    _log("6/6 rendering dashboard video")
    renderer = DashboardRenderer(court, players, ball, shots, rallies, fps, (sw, sh),
                                 names={"near": near_name, "far": far_name}, title=title)
    writer = _Writer(out / "analytics.mp4", fps)
    cap = cv2.VideoCapture(str(src))
    i = 0
    last = None
    while i < n:
        ok, frame = cap.read()
        if not ok:
            break
        last = renderer.render(frame, i)
        writer.write(last)
        i += 1
        if i % 100 == 0:
            _log(f"    frame {i}/{n}")
    cap.release()
    writer.close()
    if last is not None:
        cv2.imwrite(str(out / "final_frame.jpg"), last)
    _log(f"done in {time.time() - t0:.0f}s -> {out / 'analytics.mp4'}")
    return summary
