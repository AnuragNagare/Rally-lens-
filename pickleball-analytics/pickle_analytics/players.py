"""Player detection with YOLO pose, per-side tracking, and projection onto the court."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.signal import medfilt, savgol_filter

from .court import COURT_L, COURT_W, NET_Y, Court

# COCO keypoint indices
L_WRIST, R_WRIST, L_ANKLE, R_ANKLE = 9, 10, 15, 16
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]

SIDES = ("near", "far")


@dataclass
class PlayerTrack:
    """One player's per-frame data. NaN where the player was not seen."""
    n: int
    kpts: np.ndarray = field(init=False)      # (n, 17, 3) x, y, confidence
    bbox: np.ndarray = field(init=False)      # (n, 4) x1, y1, x2, y2
    foot_px: np.ndarray = field(init=False)   # (n, 2)
    court: np.ndarray = field(init=False)     # (n, 2) metres, smoothed
    speed: np.ndarray = field(init=False)     # (n,) km/h, smoothed
    dist: np.ndarray = field(init=False)      # (n,) cumulative metres
    seen: np.ndarray = field(init=False)      # (n,) bool, raw detection present

    def __post_init__(self):
        self.kpts = np.full((self.n, 17, 3), np.nan, dtype=np.float32)
        self.bbox = np.full((self.n, 4), np.nan, dtype=np.float32)
        self.foot_px = np.full((self.n, 2), np.nan, dtype=np.float32)
        self.court = np.full((self.n, 2), np.nan, dtype=np.float32)
        self.speed = np.zeros(self.n, dtype=np.float32)
        self.dist = np.zeros(self.n, dtype=np.float32)
        self.seen = np.zeros(self.n, dtype=bool)


def far_half_roi(court: Court, shape) -> tuple[int, int, int, int]:
    """Image rectangle around the far half of the court (where the far player is small)."""
    h, w = shape[:2]
    c = court.corners_px
    net = court.to_image([[0, NET_Y], [COURT_W, NET_Y]])
    far_y = min(c[0, 1], c[1, 1])
    depth = max(c[2, 1], c[3, 1]) - far_y
    x1 = min(net[0, 0], c[0, 0]) - 0.22 * (net[1, 0] - net[0, 0])
    x2 = max(net[1, 0], c[1, 0]) + 0.22 * (net[1, 0] - net[0, 0])
    y1 = far_y - 0.32 * depth
    y2 = max(net[:, 1]) + 0.06 * depth
    return int(max(x1, 0)), int(max(y1, 0)), int(min(x2, w)), int(min(y2, h))


def detect_people(video_path: str, weights: str = "yolo11s-pose.pt", imgsz: int = 1280,
                  conf: float = 0.2, far_roi: tuple[int, int, int, int] | None = None,
                  progress=None) -> list[dict]:
    """Run the pose model on every frame. Returns one dict per frame: boxes (k,4), kpts (k,17,3).

    The far player is only ~100 px tall in a 1080p frame and is often missed in a
    full-frame pass. If ``far_roi`` is given, that crop is run as a second, zoomed-in pass
    and its detections are added (shifted back to full-frame coordinates).
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    cap = cv2.VideoCapture(video_path)
    out = []
    i = 0

    def run(img, dx=0, dy=0):
        r = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            return np.zeros((0, 4), np.float32), np.zeros((0, 17, 3), np.float32)
        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        kpts = r.keypoints.data.cpu().numpy().astype(np.float32).copy()
        boxes[:, [0, 2]] += dx
        boxes[:, [1, 3]] += dy
        kpts[:, :, 0] += dx
        kpts[:, :, 1] += dy
        return boxes, kpts

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes, kpts = run(frame)
        if far_roi is not None:
            x1, y1, x2, y2 = far_roi
            b2, k2 = run(frame[y1:y2, x1:x2], x1, y1)
            # ignore people cut by the crop's bottom edge (they belong to the full-frame pass)
            keep = b2[:, 3] < y2 - 3
            boxes = np.concatenate([boxes, b2[keep]])
            kpts = np.concatenate([kpts, k2[keep]])
        out.append({"boxes": boxes, "kpts": kpts})
        i += 1
        if progress and i % 25 == 0:
            progress(i)
    cap.release()
    return out


def foot_point(box: np.ndarray, kpts: np.ndarray, frame_h: int) -> np.ndarray:
    """Where the player touches the ground: midpoint of the ankles, else bottom-centre of the box."""
    ankles = kpts[[L_ANKLE, R_ANKLE]]
    good = ankles[:, 2] > 0.5
    if good.all() and ankles[:, 1].max() < frame_h - 4:
        # ankle joints sit ~8% of body height above the sole; push down to the ground contact
        return np.array([ankles[:, 0].mean(), ankles[:, 1].max() + 0.04 * (box[3] - box[1])], np.float32)
    return np.array([(box[0] + box[2]) / 2, box[3]], np.float32)


def track_players(detections: list[dict], court: Court, frame_h: int, fps: float,
                  margin_x: float = 3.0, margin_y: float = 4.5,
                  static_share: float = 0.6) -> dict[str, PlayerTrack]:
    """Pick one player per half of the court in every frame, then smooth their paths.

    A detection is a candidate if its foot point lands on or near the court (this drops
    spectators, officials and ball kids). Per side we keep the candidate closest to where
    that player was last seen.
    """
    n = len(detections)
    tracks = {s: PlayerTrack(n) for s in SIDES}
    last = {s: None for s in SIDES}
    lost = {s: 0 for s in SIDES}
    raw = {s: np.full((n, 2), np.nan, dtype=np.float32) for s in SIDES}

    # Project every detection once.
    projected = []
    for det in detections:
        items = []
        for box, kp in zip(det["boxes"], det["kpts"]):
            fp = foot_point(box, kp, frame_h)
            cx, cy = court.to_court(fp)[0]
            items.append((box, kp, fp, np.array([cx, cy], np.float32)))
        projected.append(items)

    # People who never move (officials, seated ball kids) occupy the same 0.6 m cell in
    # most frames. Players do not. Mark those cells and ignore whoever stands in them.
    cell = 0.6
    counts: dict[tuple[int, int], int] = {}
    for items in projected:
        for key in {(int(np.floor(c[3][0] / cell)), int(np.floor(c[3][1] / cell))) for c in items}:
            counts[key] = counts.get(key, 0) + 1
    static = {k for k, v in counts.items() if v > static_share * n}
    static |= {(kx + dx, ky + dy) for kx, ky in list(static) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}

    for i, items in enumerate(projected):
        cands = {s: [] for s in SIDES}
        for box, kp, fp, xy in items:
            cx, cy = xy
            if not (-margin_x <= cx <= COURT_W + margin_x and -margin_y <= cy <= COURT_L + margin_y):
                continue
            if (int(np.floor(cx / cell)), int(np.floor(cy / cell))) in static:
                continue
            side = "near" if cy > NET_Y else "far"
            cands[side].append((box, kp, fp, xy))
        for s in SIDES:
            if not cands[s]:
                lost[s] += 1
                continue
            if last[s] is None:
                # first sighting: the candidate closest to the court's long axis
                best = min(cands[s], key=lambda c: abs(c[3][0] - COURT_W / 2))
            else:
                best = min(cands[s], key=lambda c: np.linalg.norm(c[3] - last[s]))
                # a jump faster than a sprint means we grabbed someone else
                max_step = 10.0 / fps * (lost[s] + 1) + 0.6
                if np.linalg.norm(best[3] - last[s]) > max_step and lost[s] < fps:
                    lost[s] += 1
                    continue
            t = tracks[s]
            t.bbox[i], t.kpts[i], t.foot_px[i] = best[0], best[1], best[2]
            t.seen[i] = True
            raw[s][i] = best[3]
            last[s] = best[3]
            lost[s] = 0

    for s in SIDES:
        _smooth_and_measure(tracks[s], raw[s], fps)
    return tracks


def _smooth_and_measure(t: PlayerTrack, raw: np.ndarray, fps: float):
    """Fill short gaps, smooth the court path, and derive speed and distance."""
    n = len(raw)
    idx = np.arange(n)
    ok = ~np.isnan(raw[:, 0])
    if ok.sum() < 5:
        return
    first, last = idx[ok][0], idx[ok][-1]
    filled = np.stack([np.interp(idx, idx[ok], raw[ok, k]) for k in range(2)], axis=1)
    filled = medfilt(filled, kernel_size=(5, 1))         # kill single-frame foot-point jumps
    win = int(round(fps * 0.5)) | 1                      # ~0.5 s window, odd
    win = max(5, min(win, (ok.sum() // 2) * 2 - 1))
    smooth = savgol_filter(filled, win, 2, axis=0)
    smooth[:first] = np.nan
    smooth[last + 1:] = np.nan
    # do not invent a path across long absences (player walked out of shot)
    seen_idx = idx[ok]
    for a, b in zip(seen_idx[:-1], seen_idx[1:]):
        if b - a > 0.3 * fps:
            smooth[a + 1:b] = np.nan
    t.court[:] = smooth

    step = np.zeros(n, dtype=np.float32)
    step[first + 1:last + 1] = np.linalg.norm(np.diff(smooth[first:last + 1], axis=0), axis=1)
    step = np.nan_to_num(step, nan=0.0)                 # no distance credited across absences
    speed_ms = step * fps
    k = max(3, int(round(fps * 0.7)) | 1)             # ~0.7 s: steps, not pose jitter
    speed_ms = np.convolve(speed_ms, np.ones(k) / k, mode="same")
    # below 0.3 m/s is pose jitter from a standing player, not movement
    moving = np.where(speed_ms > 0.3, step, 0.0)
    t.speed[:] = speed_ms * 3.6
    t.dist[:] = np.cumsum(moving)
