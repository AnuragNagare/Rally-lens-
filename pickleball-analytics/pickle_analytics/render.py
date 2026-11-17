"""Dashboard renderer: annotated video + 2D court + live stats, composed into one 1920x1080 frame."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .ball import BallTrack
from .court import (COURT_L, COURT_LINES_M, COURT_W, KITCHEN_FAR, KITCHEN_NEAR, NET_Y, NVZ, Court)
from .players import SKELETON, PlayerTrack
from .shots import Shot

# ---------------------------------------------------------------------------- palette
# Dark-surface palette. Identity colours are fixed per entity: near = blue, far = orange.
# Text always uses the ink colours; a coloured mark next to it carries identity.
PAGE = (13, 13, 13)
SURFACE = (26, 26, 25)
INK = (240, 239, 236)
INK2 = (195, 194, 183)
MUTED = (137, 135, 129)
GRID = (44, 44, 42)
AXIS = (56, 56, 53)
NEAR = (57, 135, 229)
FAR = (217, 89, 38)
BALL = (255, 214, 10)      # overlay marker only (bright so it reads on a green court)
COLOR = {"near": NEAR, "far": FAR}

W, H = 1920, 1080
PAD = 16
HEADER_H = 56
VID = (PAD, HEADER_H + 8, 1216, 684)                    # x, y, w, h
COURT_PANEL = (VID[0] + VID[2] + PAD, HEADER_H + 8, W - (VID[0] + VID[2] + PAD) - PAD, H - HEADER_H - 8 - PAD)
STAT_Y = VID[1] + VID[3] + PAD
STAT_H = H - STAT_Y - PAD
PANEL_W = 276
PANELS = {
    "near": (PAD, STAT_Y, PANEL_W, STAT_H),
    "far": (PAD + PANEL_W + PAD, STAT_Y, PANEL_W, STAT_H),
    "rally": (PAD + 2 * (PANEL_W + PAD), STAT_Y, PANEL_W, STAT_H),
    "chart": (PAD + 3 * (PANEL_W + PAD), STAT_Y, VID[2] - 3 * (PANEL_W + PAD), STAT_H),
}

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"


def bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])


def mix(c1, c2, a):
    """a = share of c1."""
    return tuple(int(round(a * x + (1 - a) * y)) for x, y in zip(c1, c2))


class Fonts:
    def __init__(self):
        self._cache = {}

    def get(self, size: int, weight: str = "Regular"):
        key = (size, weight)
        if key not in self._cache:
            for path in (FONT_DIR / f"Inter-{weight}.otf",
                         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if weight != "Regular"
                              else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
                try:
                    self._cache[key] = ImageFont.truetype(str(path), size)
                    break
                except OSError:
                    continue
            else:
                self._cache[key] = ImageFont.load_default()
        return self._cache[key]


class TextLayer:
    """Collects text draws, then applies them in one PIL pass (crisp, anti-aliased type)."""

    def __init__(self, fonts: Fonts):
        self.fonts = fonts
        self.ops = []

    def text(self, xy, s, size=16, color=INK, weight="Regular", anchor="la"):
        self.ops.append((xy, s, size, color, weight, anchor))

    def width(self, s, size=16, weight="Regular") -> int:
        return int(self.fonts.get(size, weight).getlength(s))

    def flush(self, canvas_bgr: np.ndarray) -> np.ndarray:
        if not self.ops:
            return canvas_bgr
        img = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        for xy, s, size, color, weight, anchor in self.ops:
            d.text(xy, s, font=self.fonts.get(size, weight), fill=color, anchor=anchor)
        self.ops = []
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def rounded_rect(img, x, y, w, h, color, r=8):
    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1, cv2.LINE_AA)
    for cx, cy in ((x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def pill(img, x, y, w, h, alpha=0.78):
    """Translucent dark plate so overlay text stays readable on any footage."""
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha) + np.array(bgr(PAGE)) * alpha).astype(np.uint8)


class DashboardRenderer:
    def __init__(self, court: Court, players: dict[str, PlayerTrack], ball: BallTrack,
                 shots: list[Shot], rallies: list[tuple], fps: float, src_size: tuple[int, int],
                 names: dict[str, str] | None = None, title: str = "PICKLEBALL AI ANALYTICS"):
        self.court, self.players, self.ball = court, players, ball
        self.shots, self.rallies, self.fps = shots, rallies, fps
        self.src_w, self.src_h = src_size
        self.n = len(ball.xy)
        self.names = names or {"near": "Near player", "far": "Far player"}
        self.title = title
        self.fonts = Fonts()
        self.scale = VID[2] / self.src_w

        # 2D court geometry: metres -> panel pixels
        px, py, pw, ph = COURT_PANEL
        self.m_side, self.m_end = 2.4, 2.4
        top, bottom = 56, 64
        self.ppm = min((pw - 40) / (COURT_W + 2 * self.m_side), (ph - top - bottom) / (COURT_L + 2 * self.m_end))
        self.cx0 = px + (pw - COURT_W * self.ppm) / 2
        self.cy0 = py + top + self.m_end * self.ppm
        self.heat = {s: np.zeros((ph, pw), np.float32) for s in players}
        k = int(self.ppm * 1.1) | 1
        g = cv2.getGaussianKernel(k, k / 4.5)
        self.stamp = (g @ g.T).astype(np.float32)
        self.stamp /= self.stamp.max()

        self._precompute()
        self.base = self._draw_base()

    # ------------------------------------------------------------------ helpers
    def m2p(self, x, y):
        return int(round(self.cx0 + x * self.ppm)), int(round(self.cy0 + y * self.ppm))

    def _precompute(self):
        n, fps = self.n, self.fps
        self.shot_at = np.full(n, -1, dtype=int)        # index into self.shots of the current shot
        for k, s in enumerate(self.shots):
            rally_end = self.rallies[s.rally - 1][1]
            end = s.end_frame if s.end_frame is not None else min(rally_end + int(0.8 * fps), n - 1)
            self.shot_at[s.frame:end + 1] = k
        self.rally_at = np.zeros(n, dtype=int)
        for ri, (a, b) in enumerate(self.rallies, start=1):
            self.rally_at[a:] = ri
        self.shots_by = {s: np.zeros(n, dtype=int) for s in self.players}
        for s in self.shots:
            self.shots_by[s.side][s.frame:] += 1
        self.top_speed = {s: np.maximum.accumulate(self.players[s].speed) for s in self.players}
        self.kitchen_s = {}
        for s, t in self.players.items():
            at_line = (np.abs(t.court[:, 1] - NET_Y) <= NVZ + 0.9) & ~np.isnan(t.court[:, 1])
            self.kitchen_s[s] = np.cumsum(at_line) / fps
        self.det_cum = np.cumsum(self.ball.state == 1)
        self.interp_cum = np.cumsum(self.ball.state == 2)
        vmax = max(float(max(t.speed.max() for t in self.players.values())), 1.0)
        self.chart_max = float(np.ceil(vmax / 10.0) * 10.0)

    def _draw_base(self) -> np.ndarray:
        img = np.full((H, W, 3), bgr(PAGE), np.uint8)
        t = TextLayer(self.fonts)
        # header
        t.text((PAD, HEADER_H // 2 + 2), self.title, 26, INK, "Bold", "lm")
        tw = t.width(self.title, 26, "Bold")
        t.text((PAD + tw + 28, HEADER_H // 2 + 3), "pose estimation   |   ball detection + tracking   |   court homography",
               16, MUTED, "Regular", "lm")
        # panels
        for key, (x, y, w, h) in PANELS.items():
            rounded_rect(img, x, y, w, h, bgr(SURFACE))
        x, y, w, h = COURT_PANEL
        rounded_rect(img, x, y, w, h, bgr(SURFACE))
        t.text((x + 20, y + 28), "2D COURT VIEW", 17, INK2, "SemiBold", "lm")
        t.text((x + 20 + t.width("2D COURT VIEW", 17, "SemiBold") + 12, y + 28), "top-down, from homography", 14, MUTED, "Regular", "lm")
        # stat panel titles: coloured swatch + ink text
        for key, label in (("near", self.names["near"].upper()), ("far", self.names["far"].upper())):
            x, y, w, h = PANELS[key]
            cv2.rectangle(img, (x, y), (x + w, y + 4), bgr(COLOR[key]), -1)
            rounded_rect(img, x + 18, y + 22, 14, 14, bgr(COLOR[key]), 3)
            t.text((x + 42, y + 30), label, 17, INK, "SemiBold", "lm")
        x, y, w, h = PANELS["rally"]
        cv2.rectangle(img, (x, y), (x + w, y + 4), bgr(MUTED), -1)
        t.text((x + 18, y + 30), "RALLY / BALL", 17, INK, "SemiBold", "lm")
        x, y, w, h = PANELS["chart"]
        cv2.rectangle(img, (x, y), (x + w, y + 4), bgr(MUTED), -1)
        t.text((x + 18, y + 30), "PLAYER SPEED", 17, INK, "SemiBold", "lm")
        t.text((x + 18 + t.width("PLAYER SPEED", 17, "SemiBold") + 8, y + 30), "km/h", 14, MUTED, "Regular", "lm")
        # chart legend (2 series -> legend always present)
        lx = x + w - 138
        for key, label in (("near", "Near"), ("far", "Far")):
            cv2.line(img, (lx, y + 30), (lx + 18, y + 30), bgr(COLOR[key]), 3, cv2.LINE_AA)
            t.text((lx + 24, y + 30), label, 14, INK2, "Regular", "lm")
            lx += 72
        # chart frame
        self.plot = (x + 44, y + 58, w - 44 - 18, h - 58 - 40)
        gx, gy, gw, gh = self.plot
        for frac in (0, 0.5, 1.0):
            yy = int(gy + gh - frac * gh)
            cv2.line(img, (gx, yy), (gx + gw, yy), bgr(AXIS if frac == 0 else GRID), 1)
            t.text((gx - 8, yy), f"{self.chart_max * frac:.0f}", 13, MUTED, "Regular", "rm")
        dur = self.n / self.fps
        for sec in np.arange(0, dur + 0.01, 5 if dur > 12 else 2):
            xx = int(gx + sec / dur * gw)
            t.text((xx, gy + gh + 12), f"{sec:.0f}s", 13, MUTED, "Regular", "mm")
        t.text((gx, gy + gh + 30), "ticks = shots", 12, MUTED, "Regular", "lm")
        # court legend
        x, y, w, h = COURT_PANEL
        ly = y + h - 26
        lx = x + 20
        for col, label in ((NEAR, "Near"), (FAR, "Far"), (BALL, "Ball (est.)")):
            cv2.circle(img, (lx + 6, ly), 6, bgr(col), -1, cv2.LINE_AA)
            t.text((lx + 20, ly), label, 14, INK2, "Regular", "lm")
            lx += 20 + t.width(label, 14) + 26
        t.text((x + w - 20, ly), "glow = where each player spent time", 13, MUTED, "Regular", "rm")
        return t.flush(img)

    # ------------------------------------------------------------------ video panel
    def _video_panel(self, frame: np.ndarray, i: int, t: TextLayer) -> np.ndarray:
        vx, vy, vw, vh = VID
        sc = self.scale
        img = cv2.resize(frame, (vw, vh), interpolation=cv2.INTER_AREA)

        # projected court lines (shows the homography is locked on)
        overlay = img.copy()
        for a, b in COURT_LINES_M:
            p = (self.court.to_image([a, b]) * sc).astype(int)
            cv2.line(overlay, tuple(map(int, p[0])), tuple(map(int, p[1])), (255, 255, 255), 1, cv2.LINE_AA)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

        # skeleton glow + crisp lines
        glow = np.zeros_like(img)
        for side, tr in self.players.items():
            if not tr.seen[i]:
                continue
            col = bgr(COLOR[side])
            kp = tr.kpts[i]
            for a, b in SKELETON:
                if kp[a, 2] > 0.35 and kp[b, 2] > 0.35:
                    pa = (int(kp[a, 0] * sc), int(kp[a, 1] * sc))
                    pb = (int(kp[b, 0] * sc), int(kp[b, 1] * sc))
                    cv2.line(glow, pa, pb, col, 7, cv2.LINE_AA)
        if glow.any():
            img = cv2.add(img, cv2.GaussianBlur(glow, (0, 0), 6))
        for side, tr in self.players.items():
            if not tr.seen[i]:
                continue
            col = bgr(mix(COLOR[side], (255, 255, 255), 0.55))
            kp = tr.kpts[i]
            for a, b in SKELETON:
                if kp[a, 2] > 0.35 and kp[b, 2] > 0.35:
                    pa = (int(kp[a, 0] * sc), int(kp[a, 1] * sc))
                    pb = (int(kp[b, 0] * sc), int(kp[b, 1] * sc))
                    cv2.line(img, pa, pb, col, 2, cv2.LINE_AA)
            for j in range(5, 17):
                if kp[j, 2] > 0.35:
                    cv2.circle(img, (int(kp[j, 0] * sc), int(kp[j, 1] * sc)), 3, (255, 255, 255), -1, cv2.LINE_AA)
            # ground contact marker
            fx, fy = (tr.foot_px[i] * sc).astype(int)
            bw = int((tr.bbox[i, 2] - tr.bbox[i, 0]) * sc)
            cv2.ellipse(img, (int(fx), int(fy)), (max(bw // 3, 10), max(bw // 9, 4)), 0, 0, 360, bgr(COLOR[side]), 2, cv2.LINE_AA)

        # ball trail + marker
        trail = [k for k in range(max(0, i - 12), i + 1) if self.ball.state[k] > 0]
        for k in trail[:-1] if trail and trail[-1] == i else trail:
            age = (i - k) / 12
            p = (self.ball.xy[k] * sc).astype(int)
            cv2.circle(img, (int(p[0]), int(p[1])), max(2, int(5 * (1 - age))), bgr(BALL), -1, cv2.LINE_AA)
        if self.ball.state[i] > 0:
            p = (self.ball.xy[i] * sc).astype(int)
            c = (int(p[0]), int(p[1]))
            cv2.circle(img, c, 11, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.circle(img, c, 11, bgr(BALL), 2, cv2.LINE_AA)
            if self.ball.state[i] == 1:
                cv2.circle(img, c, 4, bgr(BALL), -1, cv2.LINE_AA)

        # hit flash
        for s in self.shots:
            d = i - s.frame
            if 0 <= d <= 6 and self.ball.state[min(s.frame, self.n - 1)] > 0:
                p = (self.ball.xy[s.frame] * sc).astype(int)
                cv2.circle(img, (int(p[0]), int(p[1])), 14 + d * 6, (255, 255, 255), max(1, 3 - d // 2), cv2.LINE_AA)

        # player labels (ink text on a dark plate, colour stripe carries identity)
        for side, tr in self.players.items():
            if not tr.seen[i]:
                continue
            label = f"{self.names[side].upper()}   {tr.speed[i]:.1f} km/h"
            tw = t.width(label, 15, "SemiBold")
            bx = int((tr.bbox[i, 0] + tr.bbox[i, 2]) / 2 * sc - tw / 2 - 12)
            by = int(tr.bbox[i, 1] * sc) - 34
            bx = int(np.clip(bx, 4, vw - tw - 28))
            by = int(np.clip(by, 60, vh - 30))
            pill(img, bx, by, tw + 24, 26)
            cv2.rectangle(img, (bx, by), (bx + 4, by + 26), bgr(COLOR[side]), -1)
            t.text((vx + bx + 14, vy + by + 13), label, 15, INK, "SemiBold", "lm")

        # shot banner
        k = self.shot_at[i]
        if k >= 0:
            s = self.shots[k]
            speed = f"≈ {s.speed_kmh:.0f} km/h avg" if s.speed_kmh is not None else "rally-ending shot"
            label = f"RALLY {s.rally}  ·  SHOT {s.number_in_rally}  ·  {self.names[s.side].upper()}  ·  {speed}"
            tw = t.width(label, 22, "SemiBold")
            pill(img, 14, 14, tw + 40, 44, 0.82)
            cv2.rectangle(img, (14, 14), (20, 58), bgr(COLOR[s.side]), -1)
            t.text((vx + 36, vy + 36), label, 22, INK, "SemiBold", "lm")
        return img

    # ------------------------------------------------------------------ 2D court
    def _court_panel(self, canvas: np.ndarray, i: int, t: TextLayer):
        px, py, pw, ph = COURT_PANEL
        roi = canvas[py:py + ph, px:px + pw]

        # accumulate presence
        for side, tr in self.players.items():
            if np.isnan(tr.court[i, 0]):
                continue
            cx, cy = self.m2p(*tr.court[i])
            cx -= px
            cy -= py
            k = self.stamp.shape[0] // 2
            x0, y0, x1, y1 = cx - k, cy - k, cx + k + 1, cy + k + 1
            sx0, sy0 = max(0, -x0), max(0, -y0)
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, pw), min(y1, ph)
            if x1 > x0 and y1 > y0:
                self.heat[side][y0:y1, x0:x1] += self.stamp[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]

        # heat: one hue per player, intensity = time spent (sequential, single hue)
        out = roi.astype(np.float32)
        for side in self.players:
            hmap = self.heat[side]
            peak = max(float(hmap.max()), self.fps * 0.8)       # ~0.8 s in one spot = full strength
            a = np.clip(hmap / peak, 0, 1) ** 0.65 * 0.62
            col = np.array(bgr(COLOR[side]), np.float32)
            out = out * (1 - a[..., None]) + col * a[..., None]
        roi[:] = out.astype(np.uint8)

        # court lines
        line_col = bgr(mix(INK2, SURFACE, 0.55))
        for a, b in COURT_LINES_M:
            cv2.line(canvas, self.m2p(*a), self.m2p(*b), line_col, 2, cv2.LINE_AA)
        n0, n1 = self.m2p(-0.5, NET_Y), self.m2p(COURT_W + 0.5, NET_Y)
        cv2.line(canvas, n0, n1, bgr(INK), 3, cv2.LINE_AA)
        t.text((n1[0] + 8, n1[1]), "NET", 12, MUTED, "SemiBold", "lm")
        kx = self.m2p(COURT_W / 2, 0)[0]
        t.text((kx, self.m2p(0, (KITCHEN_FAR + NET_Y) / 2)[1]), "KITCHEN", 11, MUTED, "SemiBold", "mm")
        t.text((kx, self.m2p(0, (KITCHEN_NEAR + NET_Y) / 2)[1]), "KITCHEN", 11, MUTED, "SemiBold", "mm")

        # shot lines: hitter -> receiver. Recent shots strong, older ones fade back.
        done = [s for s in self.shots if s.frame <= i and s.receiver_xy is not None]
        ball_pos = None
        for rank, s in enumerate(done):
            age = len(done) - 1 - rank
            alpha = (1.0, 0.6, 0.38)[age] if age < 3 else 0.16
            col = bgr(mix(COLOR[s.side], SURFACE, alpha))
            a = np.array(s.hitter_xy, float)
            b = np.array(s.receiver_xy, float)
            frac = float(np.clip((i - s.frame) / max(s.end_frame - s.frame, 1), 0, 1))
            cur = a + (b - a) * frac
            cv2.line(canvas, self.m2p(*a), self.m2p(*cur), col, 2, cv2.LINE_AA)
            cv2.circle(canvas, self.m2p(*a), 5, bgr(SURFACE), -1, cv2.LINE_AA)
            cv2.circle(canvas, self.m2p(*a), 4, col, -1, cv2.LINE_AA)
            if frac < 1.0:
                ball_pos = cur

        # short movement tail + player markers
        for side, tr in self.players.items():
            lo = max(0, i - int(self.fps * 1.5))
            pts = [self.m2p(*p) for p in tr.court[lo:i + 1] if not np.isnan(p[0])]
            if len(pts) > 1:
                cv2.polylines(canvas, [np.array(pts, np.int32)], False, bgr(COLOR[side]), 2, cv2.LINE_AA)
            if not np.isnan(tr.court[i, 0]):
                c = self.m2p(*tr.court[i])
                cv2.circle(canvas, c, 14, bgr(SURFACE), -1, cv2.LINE_AA)
                cv2.circle(canvas, c, 12, bgr(COLOR[side]), -1, cv2.LINE_AA)
                t.text(c, "N" if side == "near" else "F", 14, (255, 255, 255), "Bold", "mm")
        if ball_pos is not None:
            c = self.m2p(*ball_pos)
            cv2.circle(canvas, c, 8, bgr(SURFACE), -1, cv2.LINE_AA)
            cv2.circle(canvas, c, 6, bgr(BALL), -1, cv2.LINE_AA)

    # ------------------------------------------------------------------ stats
    def _stat_rows(self, t: TextLayer, panel, rows):
        x, y, w, h = panel
        row_h = (h - 58) / len(rows)
        for r, (label, value) in enumerate(rows):
            cy = int(y + 58 + row_h * (r + 0.5))
            t.text((x + 18, cy), label, 15, INK2, "Regular", "lm")
            t.text((x + w - 18, cy), value, 21, INK, "SemiBold", "rm")

    def _stats(self, canvas: np.ndarray, i: int, t: TextLayer):
        for side, tr in self.players.items():
            y = tr.court[i, 1]
            net = "–" if np.isnan(y) else f"{abs(y - NET_Y):.1f} m"
            self._stat_rows(t, PANELS[side], [
                ("Distance", f"{tr.dist[i]:.1f} m"),
                ("Speed", f"{tr.speed[i]:.1f} km/h"),
                ("Top speed", f"{self.top_speed[side][i]:.1f} km/h"),
                ("Shots", f"{self.shots_by[side][i]}"),
                ("From net", net),
                ("At kitchen line", f"{self.kitchen_s[side][i]:.1f} s"),
            ])
        past = [s for s in self.shots if s.frame <= i]
        speeds = [s.speed_kmh for s in past if s.speed_kmh is not None]
        k = self.shot_at[i]
        cur = self.shots[k] if k >= 0 else None
        this = "–" if cur is None else ("–" if cur.speed_kmh is None else f"≈ {cur.speed_kmh:.0f} km/h")
        self._stat_rows(t, PANELS["rally"], [
            ("Rally", f"{self.rally_at[i]} of {len(self.rallies)}" if self.rally_at[i] else "–"),
            ("Shots (total)", f"{len(past)}"),
            ("This shot (avg)", this),
            ("Avg shot speed", f"≈ {np.mean(speeds):.0f} km/h" if speeds else "–"),
            ("Fastest shot", f"≈ {max(speeds):.0f} km/h" if speeds else "–"),
            ("Ball detected", f"{self.det_cum[i]} / {i + 1} fr"),
        ])
        # speed chart, revealed up to the current frame
        gx, gy, gw, gh = self.plot
        xs = gx + np.arange(i + 1) / max(self.n - 1, 1) * gw
        for side, tr in self.players.items():
            v = tr.speed[:i + 1].astype(float)
            valid = ~np.isnan(tr.court[:i + 1, 0])
            ys = gy + gh - np.clip(v / self.chart_max, 0, 1) * gh
            pts = np.stack([xs, ys], axis=1).astype(np.int32)
            # break the line where the player was not in shot
            start = None
            for j in range(i + 2):
                ok = j <= i and valid[j]
                if ok and start is None:
                    start = j
                elif not ok and start is not None:
                    if j - start > 1:
                        cv2.polylines(canvas, [pts[start:j]], False, bgr(COLOR[side]), 2, cv2.LINE_AA)
                    start = None
            if valid[i]:
                c = (int(pts[i, 0]), int(pts[i, 1]))
                cv2.circle(canvas, c, 6, bgr(SURFACE), -1, cv2.LINE_AA)
                cv2.circle(canvas, c, 4, bgr(COLOR[side]), -1, cv2.LINE_AA)
        for s in self.shots:
            if s.frame <= i:
                xx = int(gx + s.frame / max(self.n - 1, 1) * gw)
                cv2.line(canvas, (xx, gy + gh + 1), (xx, gy + gh + 5), bgr(COLOR[s.side]), 2)

    # ------------------------------------------------------------------ frame
    def render(self, frame: np.ndarray, i: int) -> np.ndarray:
        canvas = self.base.copy()
        t = TextLayer(self.fonts)
        vx, vy, vw, vh = VID
        canvas[vy:vy + vh, vx:vx + vw] = self._video_panel(frame, i, t)
        self._court_panel(canvas, i, t)
        self._stats(canvas, i, t)
        t.text((W - PAD, HEADER_H // 2 + 3), f"frame {i + 1}/{self.n}     {i / self.fps:5.2f} s", 16, INK2, "Regular", "rm")
        return t.flush(canvas)
