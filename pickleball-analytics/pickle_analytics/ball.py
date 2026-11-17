"""Ball tracking for a fixed camera.

Two detectors share one tracker:

* ``detect_candidates_classical`` - no training needed. A pickleball is a small, bright,
  yellow-green blob that is absent from the background plate. Fast and transparent, but it
  only works when the camera does not move.
* ``detect_candidates_yolo`` - drop-in replacement once you have trained your own ball
  detector (see README, "Training a custom ball model").

Both return, per frame, an array of candidates (x, y, score). ``track_ball`` then links
candidates into smooth tracklets, throws away static or isolated ones, and interpolates
short gaps. The result marks each frame as detected, interpolated or missing.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .court import Court


@dataclass
class BallTrack:
    xy: np.ndarray        # (n, 2) pixels, NaN when unknown
    state: np.ndarray     # (n,) 0 = missing, 1 = detected, 2 = interpolated

    @property
    def detected(self) -> np.ndarray:
        return self.state == 1

    @property
    def known(self) -> np.ndarray:
        return self.state > 0


def play_region_mask(court: Court, shape, side_pad: float = 0.35, top_pad: float = 0.9) -> np.ndarray:
    """Image region where the ball can plausibly be: the court quad, widened, plus the air above it."""
    h, w = shape[:2]
    c = court.corners_px.copy()
    cx = c[:, 0].mean()
    c[:, 0] = cx + (c[:, 0] - cx) * (1 + side_pad)
    far_y = min(c[0, 1], c[1, 1])
    near_y = max(c[2, 1], c[3, 1])
    lift = (near_y - far_y) * top_pad
    poly = np.array([[c[3, 0], h], [c[3, 0], near_y - lift * 0.6], [c[0, 0], far_y - lift * 0.25],
                     [c[1, 0], far_y - lift * 0.25], [c[2, 0], near_y - lift * 0.6], [c[2, 0], h]], np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def detect_candidates_classical(video_path: str, background: np.ndarray, region: np.ndarray,
                                min_area: int = 3, max_area: int = 900, progress=None) -> list[np.ndarray]:
    """Per frame: (k, 3) array of x, y, score for ball-like blobs."""
    bg = background.astype(np.int16)
    cap = cv2.VideoCapture(video_path)
    out = []
    i = 0
    kernel = np.ones((3, 3), np.uint8)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        f = frame.astype(np.int16)
        b, g, r = f[..., 0], f[..., 1], f[..., 2]
        brighter = (g - bg[..., 1]) > 45                 # ball is brighter than whatever is behind it
        # optic-yellow: green highest, red clearly above blue. White kit has r ~ g ~ b,
        # skin has r > g, court-edge blends have r ~ b - all three fail this test.
        yellow_green = ((g - b) > 40) & ((r - b) > 12) & (g >= r - 8) & (g > 170)
        mask = (brighter & yellow_green).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, region)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        cands = []
        for k in range(1, n):
            area = stats[k, cv2.CC_STAT_AREA]
            bw, bh = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            if not (min_area <= area <= max_area):
                continue
            if max(bw, bh) > 4.5 * min(bw, bh) + 4:      # long streaks are not a ball
                continue
            cands.append((cents[k][0], cents[k][1], float(area)))
        out.append(np.array(cands, dtype=np.float32).reshape(-1, 3))
        i += 1
        if progress and i % 100 == 0:
            progress(i)
    cap.release()
    return out


def detect_candidates_yolo(video_path: str, weights: str, imgsz: int = 1280, conf: float = 0.15,
                           progress=None) -> list[np.ndarray]:
    """Same output as the classical detector, from a YOLO model trained on a 'ball' class."""
    from ultralytics import YOLO

    model = YOLO(weights)
    cap = cv2.VideoCapture(video_path)
    out = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            xywh = r.boxes.xywh.cpu().numpy()
            sc = r.boxes.conf.cpu().numpy()
            out.append(np.column_stack([xywh[:, 0], xywh[:, 1], sc]).astype(np.float32))
        else:
            out.append(np.zeros((0, 3), np.float32))
        i += 1
        if progress and i % 25 == 0:
            progress(i)
    cap.release()
    return out


def _drop_static(cands: list[np.ndarray], radius: float = 3.5, window: int = 12, min_hits: int = 3):
    """Remove blobs that sit still (crowd, signage flicker): same spot in several nearby frames."""
    n = len(cands)
    kept = []
    for i, c in enumerate(cands):
        if len(c) == 0:
            kept.append(c)
            continue
        hits = np.zeros(len(c), dtype=int)
        for j in range(max(0, i - window), min(n, i + window + 1)):
            if abs(j - i) < 3 or len(cands[j]) == 0:
                continue
            d = np.linalg.norm(c[:, None, :2] - cands[j][None, :, :2], axis=2)
            hits += (d.min(axis=1) < radius)
        kept.append(c[hits < min_hits])
    return kept


def suppress_body_candidates(cands: list[np.ndarray], detections: list[dict],
                             head_r: float = 0.08, foot_r: float = 0.07) -> list[np.ndarray]:
    """Drop candidates sitting on a face or a shoe (sunglasses, visors and shoe trims are
    often ball-coloured). Radii are fractions of the person's box height, centred on the
    pose keypoints, and kept small so a ball passing close to a player survives.
    """
    out = []
    for c, det in zip(cands, detections):
        if len(c) == 0 or len(det["boxes"]) == 0:
            out.append(c)
            continue
        keep = np.ones(len(c), dtype=bool)
        for box, kp in zip(det["boxes"], det["kpts"]):
            h = max(box[3] - box[1], 1.0)
            for joints, r in (((0, 1, 2, 3, 4), head_r), ((15, 16), foot_r)):
                pts = kp[list(joints)]
                pts = pts[pts[:, 2] > 0.3][:, :2]
                if len(pts) == 0:
                    continue
                d = np.linalg.norm(c[:, None, :2] - pts[None], axis=2).min(axis=1)
                keep &= d > r * h
        out.append(c[keep])
    return out


def _relative_to_box(p: np.ndarray, boxes: np.ndarray, pad: float = 0.12):
    """Position of p inside the smallest person box containing it, in box-height units."""
    best = None
    for x1, y1, x2, y2 in boxes:
        h = max(y2 - y1, 1.0)
        px, py = pad * (x2 - x1), pad * h
        if x1 - px <= p[0] <= x2 + px and y1 - py <= p[1] <= y2 + py:
            if best is None or h < best[2]:
                best = ((p[0] - (x1 + x2) / 2) / h, (p[1] - (y1 + y2) / 2) / h, h)
    return None if best is None else best[:2]


def _rides_on_a_body(frames, pts, person_boxes) -> bool:
    """True for a face, shoe or shirt logo: it stays inside a person's box AND barely moves
    within that box. A real ball crosses the box quickly, so its relative position sweeps."""
    rel = [_relative_to_box(q, person_boxes[f]) for f, q in zip(frames, pts)]
    inside = [r for r in rel if r is not None]
    if len(inside) < 0.8 * len(rel) or len(inside) < 2:
        return False
    if len(inside) < 5:
        return False
    r = np.array(inside)
    sweep = np.linalg.norm(r.max(axis=0) - r.min(axis=0))
    return sweep < 0.15                           # box heights


def track_ball(cands: list[np.ndarray], fps: float, frame_h: int,
               person_boxes: list[np.ndarray] | None = None,
               max_gap: int = 8, max_interp_s: float = 0.8) -> BallTrack:
    """Link candidates into tracklets, keep the plausible ones, interpolate short gaps.

    ``person_boxes`` (per frame, (k, 4)) lets the tracker reject things that ride along
    with a body - a cap, a shoe, a paddle edge - which otherwise look like a slow ball.
    """
    n = len(cands)
    cands = _drop_static(cands)
    max_step = 0.11 * frame_h * 30.0 / fps        # px per frame a ball may travel (near camera)

    # --- 1. greedy tracklets with constant-velocity prediction -------------------------
    tracklets: list[dict] = []
    active: list[dict] = []
    for i in range(n):
        used = np.zeros(len(cands[i]), dtype=bool)
        still_active = []
        # extend longer tracklets first so the real ball gets first pick
        for t in sorted(active, key=lambda t: -len(t["f"])):
            gap = i - t["f"][-1]
            if gap > max_gap:
                tracklets.append(t)
                continue
            if len(cands[i]):
                if len(t["f"]) >= 2:
                    # velocity over the last few points, not just the last pair: broadcast
                    # footage often contains repeated frames, which read as "ball stopped"
                    k = min(len(t["f"]), 4)
                    v = (t["p"][-1] - t["p"][-k]) / (t["f"][-1] - t["f"][-k])
                    pred = t["p"][-1] + v * gap
                    gate = 0.5 * np.linalg.norm(v) * gap + 15 * gap + 10
                else:
                    pred = t["p"][-1]
                    gate = max_step * gap
                d = np.linalg.norm(cands[i][:, :2] - pred, axis=1)
                d[used] = np.inf
                k = int(np.argmin(d))
                if d[k] < min(gate, max_step * gap):
                    t["f"].append(i)
                    t["p"].append(cands[i][k, :2].copy())
                    used[k] = True
            still_active.append(t)
        for k in np.where(~used)[0]:
            still_active.append({"f": [i], "p": [cands[i][k, :2].copy()]})
        active = still_active
    tracklets += active

    # --- 2. keep tracklets that look like a flying ball ------------------------------------
    good = []
    for t in tracklets:
        p = np.array(t["p"])
        if len(p) < 3:
            continue
        travel = np.linalg.norm(p[-1] - p[0])
        path = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
        if travel < 12 or path < 15:
            continue
        # jitter check: a real trajectory is mostly straight over a few frames
        if path > 4.0 * travel + 40:
            continue
        if person_boxes is not None and _rides_on_a_body(t["f"], p, person_boxes):
            continue
        t["score"] = path          # distance flown: a ball covers ground, body jitter does not
        good.append(t)

    # --- 3. one ball per frame: best chain of tracklets (dynamic programme) -----------------
    # A chain may only step from one tracklet to a later one if the ball could physically
    # cover the distance in the time between them. Each tracklet is worth its number of
    # detections plus a bonus for distance flown; each join costs a little per pixel jumped.
    good.sort(key=lambda t: (t["f"][0], t["f"][-1]))
    m = len(good)
    xy = np.full((n, 2), np.nan, dtype=np.float32)
    if m:
        per_pt = [(len(t["f"]) + 0.05 * t["score"]) / len(t["f"]) for t in good]
        best = np.array([per_pt[k] * len(good[k]["f"]) for k in range(m)], dtype=float)
        prev = np.full(m, -1, dtype=int)
        cut = np.zeros(m, dtype=int)             # first point of tracklet j that the chain uses
        for j in range(m):
            fj = good[j]["f"]
            for i in range(j):
                ei = good[i]["f"][-1]
                if ei >= fj[-1]:
                    continue
                # if the two overlap in time, join at j's first point after i ends
                k = int(np.searchsorted(fj, ei, side="right"))
                gap = fj[k] - ei
                jump = float(np.linalg.norm(np.asarray(good[j]["p"][k]) - np.asarray(good[i]["p"][-1])))
                if jump > max_step * gap + 20:
                    continue
                cand_val = best[i] + per_pt[j] * (len(fj) - k) - 0.02 * jump
                if cand_val > best[j]:
                    best[j], prev[j], cut[j] = cand_val, i, k
        j = int(np.argmax(best))
        while j >= 0:
            for f, q in list(zip(good[j]["f"], good[j]["p"]))[cut[j]:]:
                xy[f] = q
            j = prev[j]
    state = (~np.isnan(xy[:, 0])).astype(np.uint8)

    # --- 5. interpolate short gaps ------------------------------------------------------
    idx = np.where(state == 1)[0]
    max_interp = int(round(max_interp_s * fps))
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a
        if 1 < gap <= max_interp:
            for f in range(a + 1, b):
                w = (f - a) / gap
                xy[f] = (1 - w) * xy[a] + w * xy[b]
                state[f] = 2
    return BallTrack(xy=xy, state=state)
