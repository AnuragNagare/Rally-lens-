"""Shot (hit) detection and per-shot speed.

During a rally the ball shuttles between the two players. Define

    p(t) = d_far / (d_far + d_near)

where d_far / d_near are the image distances from the ball to each player's body.
p is ~0 when the ball is at the far player and ~1 when it is at the near player, so every
hit is a turning point of p(t): a peak for the near player, a trough for the far player.
Bounces barely move p, so they are ignored without needing a bounce model. A turning
point only counts if the ball is actually within that player's reach at that moment.

Speed: the homography is only valid on the ground and the ball is in the air, so we do
NOT project the ball. A shot's speed is the ground distance between the hitter (at the
hit) and the receiver (at the next hit) divided by the time between the two hits. That is
an average over the whole flight including any bounce - lower than a radar-gun reading.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .ball import BallTrack
from .court import KITCHEN_FAR, KITCHEN_NEAR, NET_Y
from .players import PlayerTrack


@dataclass
class Shot:
    index: int               # 1-based over the whole clip
    rally: int               # 1-based rally number
    number_in_rally: int
    frame: int
    time_s: float
    side: str                # "near" or "far" (who hit it)
    zone: str                # where the hitter stood: kitchen / mid-court / baseline
    hitter_xy: tuple         # court metres
    receiver_xy: tuple | None = None
    end_frame: int | None = None
    distance_m: float | None = None
    speed_kmh: float | None = None

    def to_dict(self):
        return asdict(self)


def _fill(a: np.ndarray) -> np.ndarray:
    """Linear fill of NaNs (edges held)."""
    a = a.copy()
    idx = np.arange(len(a))
    for k in range(a.shape[1]):
        ok = ~np.isnan(a[:, k])
        if ok.any():
            a[:, k] = np.interp(idx, idx[ok], a[ok, k])
    return a


def _box_distance(pt, box) -> float:
    """Distance from a point to a box, in units of box height (0 = inside)."""
    x1, y1, x2, y2 = box
    dx = max(x1 - pt[0], 0, pt[0] - x2)
    dy = max(y1 - pt[1], 0, pt[1] - y2)
    return float(np.hypot(dx, dy) / max(y2 - y1, 1))


def zone_of(side: str, y: float) -> str:
    d = abs(y - NET_Y)                       # metres from the net
    if d <= 2.13 + 0.9:
        return "kitchen line"
    if d <= 4.6:
        return "mid-court"
    return "baseline"


def find_rallies(ball: BallTrack, fps: float, gap_s: float = 1.2, min_len_s: float = 0.8):
    """Split the clip into rallies: stretches where the ball is visible with no long gap."""
    known = np.where(ball.known)[0]
    if len(known) == 0:
        return []
    rallies, start, prev = [], known[0], known[0]
    for f in known[1:]:
        if f - prev > gap_s * fps:
            rallies.append((start, prev))
            start = f
        prev = f
    rallies.append((start, prev))
    return [(a, b) for a, b in rallies if (b - a) >= min_len_s * fps]


def detect_shots(ball: BallTrack, players: dict[str, PlayerTrack], fps: float,
                 prominence: float = 0.22, reach: float = 0.75,
                 min_gap_s: float = 0.2) -> tuple[list[Shot], list[tuple]]:
    boxes = {s: _fill(players[s].bbox) for s in players}
    centre = {s: np.stack([(boxes[s][:, 0] + boxes[s][:, 2]) / 2,
                           (boxes[s][:, 1] + boxes[s][:, 3]) / 2], axis=1) for s in players}
    court = {s: _fill(players[s].court) for s in players}

    d_far = np.linalg.norm(ball.xy - centre["far"], axis=1)
    d_near = np.linalg.norm(ball.xy - centre["near"], axis=1)
    p_all = d_far / (d_far + d_near + 1e-6)

    rallies = find_rallies(ball, fps)
    min_sep = max(2, int(min_gap_s * fps))
    hits: list[tuple[int, str, int]] = []           # frame, side, rally
    for ri, (a, b) in enumerate(rallies, start=1):
        seg = p_all[a:b + 1].copy()
        idx = np.arange(len(seg))
        ok = ~np.isnan(seg)
        seg = np.interp(idx, idx[ok], seg[ok])
        if len(seg) >= 7:
            seg = savgol_filter(seg, 7, 2)
        # pad the start with a neutral value so a serve (ball starts at a player) is a peak
        padded = np.concatenate([[0.5, 0.5], seg])
        cand = []
        pk, _ = find_peaks(padded, prominence=prominence, distance=min_sep)
        cand += [(a + max(int(i) - 2, 0), "near") for i in pk if padded[i] > 0.55]
        pk, _ = find_peaks(-padded, prominence=prominence, distance=min_sep)
        cand += [(a + max(int(i) - 2, 0), "far") for i in pk if padded[i] < 0.45]
        cand.sort()

        rally_hits: list[tuple[int, str, int, float]] = []
        for f, side in cand:
            # the ball must actually be within the player's reach around that moment
            win = [g for g in range(max(a, f - 3), min(b, f + 3) + 1) if ball.state[g] > 0]
            if not win:
                continue
            d = min(_box_distance(ball.xy[g], boxes[side][g]) for g in win)
            if d > reach:
                continue
            if rally_hits and rally_hits[-1][1] == side:
                # two turning points in a row for one player: keep the closer contact
                if d < rally_hits[-1][3]:
                    rally_hits[-1] = (f, side, ri, d)
                continue
            rally_hits.append((f, side, ri, d))
        # A ball knocked back to the server between points is a real contact but not a rally
        # shot. Tell-tale: the opponent is not on court yet. Drop such hits from the start.
        while rally_hits:
            f, side, _, _ = rally_hits[0]
            other = "near" if side == "far" else "far"
            lo, hi = max(0, f - 3), min(len(ball.xy), f + 4)
            if players[other].seen[lo:hi].any():
                break
            rally_hits.pop(0)
        hits += [h[:3] for h in rally_hits]

    shots: list[Shot] = []
    counter = {}
    for k, (f, side, ri) in enumerate(hits):
        counter[ri] = counter.get(ri, 0) + 1
        hx, hy = court[side][f]
        shot = Shot(index=k + 1, rally=ri, number_in_rally=counter[ri], frame=int(f),
                    time_s=round(f / fps, 3), side=side, zone=zone_of(side, hy),
                    hitter_xy=(round(float(hx), 2), round(float(hy), 2)))
        if k + 1 < len(hits) and hits[k + 1][2] == ri:
            f2, side2, _ = hits[k + 1]
            rx, ry = court[side2][f2]
            dist = float(np.hypot(rx - hx, ry - hy))
            dt = (f2 - f) / fps
            shot.receiver_xy = (round(float(rx), 2), round(float(ry), 2))
            shot.end_frame = int(f2)
            shot.distance_m = round(dist, 2)
            shot.speed_kmh = round(dist / dt * 3.6, 1) if dt > 0 else None
        shots.append(shot)
    return shots, rallies
