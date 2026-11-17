"""Court model, automatic line detection and pixel <-> metre homography.

Court coordinates (metres): x across the court 0..6.10, y along it 0..13.41.
y = 0 is the FAR baseline (top of the image), y = 13.41 the NEAR baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

COURT_W = 6.10          # sideline to sideline
COURT_L = 13.41         # baseline to baseline
NET_Y = COURT_L / 2
NVZ = 2.13              # non-volley zone ("kitchen") depth each side of the net
KITCHEN_FAR = NET_Y - NVZ
KITCHEN_NEAR = NET_Y + NVZ

# Corner order used everywhere: far-left, far-right, near-right, near-left
COURT_CORNERS_M = np.array(
    [[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]], dtype=np.float32
)

# Every painted line, as segments in metres (used for drawing and for scoring a calibration)
COURT_LINES_M = [
    ((0, 0), (COURT_W, 0)),                            # far baseline
    ((0, COURT_L), (COURT_W, COURT_L)),                # near baseline
    ((0, 0), (0, COURT_L)),                            # left sideline
    ((COURT_W, 0), (COURT_W, COURT_L)),                # right sideline
    ((0, KITCHEN_FAR), (COURT_W, KITCHEN_FAR)),        # far kitchen line
    ((0, KITCHEN_NEAR), (COURT_W, KITCHEN_NEAR)),      # near kitchen line
    ((COURT_W / 2, 0), (COURT_W / 2, KITCHEN_FAR)),    # far centre line
    ((COURT_W / 2, KITCHEN_NEAR), (COURT_W / 2, COURT_L)),  # near centre line
]


def background_frame(video_path: str, samples: int = 45) -> np.ndarray:
    """Per-pixel median of evenly spaced frames: players and ball vanish, the court stays."""
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in np.linspace(0, max(n - 1, 0), samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not read frames from {video_path}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def white_line_mask(bgr: np.ndarray) -> np.ndarray:
    """Bright, unsaturated, thin structures (painted lines)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 2] > 170) & (hsv[..., 1] < 70)).astype(np.uint8) * 255
    # keep thin structures only: remove anything that survives a large opening (shirts, boards)
    blobs = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    return cv2.subtract(mask, cv2.dilate(blobs, np.ones((9, 9), np.uint8)))


def _line_through(p1, p2):
    """Homogeneous line through two points."""
    return np.cross([p1[0], p1[1], 1.0], [p2[0], p2[1], 1.0])


def _intersect(l1, l2):
    p = np.cross(l1, l2)
    return p[:2] / p[2]


def _merge_segments(segs, horizontal: bool):
    """Group roughly collinear Hough segments. Returns dicts with a fitted line and pixel extent."""
    groups: list[dict] = []
    for x1, y1, x2, y2 in segs:
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        length = float(np.hypot(x2 - x1, y2 - y1))
        placed = False
        for g in groups:
            dang = min(abs(ang - g["ang"]), 180 - abs(ang - g["ang"]))
            if dang > 3:
                continue
            # perpendicular distance of this segment's midpoint to the group's line
            a, b, c = g["line"]
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if abs(a * mx + b * my + c) / np.hypot(a, b) < 8:
                g["pts"] += [(x1, y1), (x2, y2)]
                g["len"] += length
                placed = True
                break
        if not placed:
            groups.append({"ang": ang, "line": _line_through((x1, y1), (x2, y2)),
                           "pts": [(x1, y1), (x2, y2)], "len": length})
    for g in groups:
        pts = np.array(g["pts"], dtype=np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        g["line"] = _line_through((x0, y0), (x0 + vx, y0 + vy))
        g["xmin"], g["ymin"] = pts.min(axis=0)
        g["xmax"], g["ymax"] = pts.max(axis=0)
        g["span"] = (g["xmax"] - g["xmin"]) if horizontal else (g["ymax"] - g["ymin"])
    return groups


def detect_court_corners(background: np.ndarray) -> np.ndarray:
    """Find the four court corners (far-left, far-right, near-right, near-left) in pixels.

    Assumes the usual broadcast / club view: camera behind one baseline, baselines roughly
    horizontal in the image, sidelines converging upward.
    """
    h, w = background.shape[:2]
    mask = white_line_mask(background)
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=60,
                           minLineLength=int(0.06 * w), maxLineGap=int(0.02 * w))
    if segs is None:
        raise RuntimeError("No lines found; use tools/pick_court_points.py to calibrate by hand.")
    segs = np.asarray(segs).reshape(-1, 4).astype(float)  # OpenCV 4 returns (N,1,4), OpenCV 5 (N,4)
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    horiz = segs[(ang < 6) | (ang > 174)]
    slant = segs[(ang > 20) & (ang < 160) & (np.abs(ang - 90) > 4)]

    hgroups = _merge_segments(horiz, horizontal=True)
    sgroups = _merge_segments(slant, horizontal=False)

    # Sidelines: the tallest slanted line leaning each way (left sideline leans "/", right "\").
    left = [g for g in sgroups if g["ang"] > 90]
    right = [g for g in sgroups if g["ang"] < 90]
    if not left or not right:
        raise RuntimeError("Sidelines not found; calibrate by hand.")
    left_side = max(left, key=lambda g: g["span"])
    right_side = max(right, key=lambda g: g["span"])

    # Near baseline: the widest horizontal line in the lower half.
    lower = [g for g in hgroups if (g["ymin"] + g["ymax"]) / 2 > 0.5 * h]
    if not lower:
        raise RuntimeError("Near baseline not found; calibrate by hand.")
    near_base = max(lower, key=lambda g: g["span"])

    # Far baseline: the horizontal line where the sidelines stop at the top.
    top_y = min(left_side["ymin"], right_side["ymin"])
    upper = [g for g in hgroups if (g["ymin"] + g["ymax"]) / 2 < 0.5 * h and g["span"] > 0.1 * w]
    if not upper:
        raise RuntimeError("Far baseline not found; calibrate by hand.")
    far_base = min(upper, key=lambda g: abs((g["ymin"] + g["ymax"]) / 2 - top_y))

    corners = np.array([
        _intersect(far_base["line"], left_side["line"]),
        _intersect(far_base["line"], right_side["line"]),
        _intersect(near_base["line"], right_side["line"]),
        _intersect(near_base["line"], left_side["line"]),
    ], dtype=np.float32)
    return corners


class Court:
    """Homography wrapper: image pixels <-> court metres (valid for points ON the ground)."""

    def __init__(self, corners_px: np.ndarray):
        self.corners_px = np.asarray(corners_px, dtype=np.float32).reshape(4, 2)
        self.H = cv2.getPerspectiveTransform(self.corners_px, COURT_CORNERS_M)
        self.H_inv = np.linalg.inv(self.H)

    def to_court(self, pts_px) -> np.ndarray:
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_image(self, pts_m) -> np.ndarray:
        pts = np.asarray(pts_m, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)

    def line_fit_score(self, background: np.ndarray) -> float:
        """Share of the projected court lines that land on white paint (1.0 = perfect)."""
        mask = cv2.dilate(white_line_mask(background), np.ones((7, 7), np.uint8))
        h, w = mask.shape
        hits = total = 0
        for a, b in COURT_LINES_M:
            pts = self.to_image(np.linspace(a, b, 80))
            for x, y in pts:
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < w and 0 <= yi < h:
                    total += 1
                    hits += mask[yi, xi] > 0
        return hits / max(total, 1)

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps({"corners_px": self.corners_px.tolist(),
                                          "order": "far-left, far-right, near-right, near-left"}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Court":
        return cls(np.array(json.loads(Path(path).read_text())["corners_px"], dtype=np.float32))
