"""Anodite energy effects: particle system, orbs, beams, trails, whips, shockwaves, projectiles.

The engine is driven by gesture states (see gesture_recognition.py). It exposes:
  * ``update``         - advance simulation and spawn effects from gestures
  * ``draw``           - draw crisp elements (orbs, lines, rings) onto an additive RGB surface
  * ``particle_buffer``- accumulate particles into a low-resolution float buffer (bloom source)
"""
import math
from collections import deque

import numpy as np
import pygame

import config
from gesture_recognition import (APART, FIST, NONE, OPEN_PALM, POINT, RAISED,
                                 TOGETHER)

TAU = math.tau

# ---------------------------------------------------------------- colour helpers


def _build_lut(n=64):
    stops = np.array(config.PARTICLE_GRADIENT, dtype=np.float32)
    xs = np.linspace(0, 1, len(stops))
    t = np.linspace(0, 1, n)
    return np.stack([np.interp(t, xs, stops[:, c]) for c in range(3)], axis=1).astype(np.float32)


GRADIENT_LUT = _build_lut()

# (width multiplier, colour) from outermost glow to white-hot core
LAYERS = [(3.0, (70, 15, 150)), (1.8, (210, 30, 180)), (1.0, (255, 120, 225)), (0.4, (255, 240, 255))]


def smooth_path(p, sub=3):
    """Catmull-Rom interpolation through an (n, 2) array."""
    n = len(p)
    if n < 3:
        return p
    P = np.vstack([p[0], p, p[-1]])
    p0, p1, p2, p3 = P[:-3], P[1:-2], P[2:-1], P[3:]
    t = np.linspace(0, 1, sub, endpoint=False)[:, None, None]
    seg = 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)
    return np.vstack([seg.transpose(1, 0, 2).reshape(-1, 2), p[-1:]])


def line_segments(pts, taper=True, chunks=5):
    """Split a tail -> head polyline into tapered chunks: [(points, brightness factor)]."""
    n = len(pts)
    if n < 2:
        return []
    chunks = min(chunks, n - 1) if taper else 1
    idx = np.linspace(0, n - 1, chunks + 1).astype(int)
    segs = []
    for c in range(chunks):
        seg = pts[idx[c]:min(idx[c + 1] + 1, n)]
        if len(seg) >= 2:
            segs.append((seg.tolist(), (c + 1) / chunks if taper else 1.0))
    return segs


def flush_lines(surf, batch):
    """Draw queued glow polylines layer-major so brighter cores always overwrite dim halos."""
    for wm, col in LAYERS:
        for segs, width, strength in batch:
            for seg_l, f in segs:
                k = min(1.0, f * strength)
                color = (int(col[0] * k), int(col[1] * k), int(col[2] * k))
                if max(color) < 14:
                    continue
                w = max(1, int(width * f * wm))
                pygame.draw.lines(surf, color, False, seg_l, w)
                if w > 3:
                    pygame.draw.circle(surf, color, seg_l[-1], w // 2)


def lightning_points(a, b, rng, disp=0.22, depth=4):
    p = np.array([a, b], dtype=np.float64)
    for _ in range(depth):
        mid = (p[:-1] + p[1:]) / 2
        d = p[1:] - p[:-1]
        L = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
        perp = np.concatenate([-d[:, 1:2], d[:, 0:1]], axis=1) / L
        mid = mid + perp * rng.normal(size=(len(mid), 1)) * L * disp
        new = np.empty((len(p) + len(mid), 2))
        new[0::2], new[1::2] = p, mid
        p = new
    return p


# ---------------------------------------------------------------- procedural sprites

_BASE = 256


def _grid():
    ys, xs = np.mgrid[0:_BASE, 0:_BASE].astype(np.float32)
    c = (_BASE - 1) / 2
    x, y = (xs - c) / c, (ys - c) / c
    return np.sqrt(x * x + y * y), np.arctan2(y, x)


def _to_surface(rgb):
    return pygame.surfarray.make_surface(np.clip(rgb, 0, 255).astype(np.uint8).swapaxes(0, 1))


def _make_orb():
    r, _ = _grid()
    f = np.clip(1 - r, 0, 1)
    core, mid, halo = f ** 6, f ** 2.2, f ** 1.1
    rgb = (core[..., None] * np.array([255, 245, 255]) + mid[..., None] * np.array([255, 70, 200])
           + halo[..., None] * np.array([110, 25, 190]) * 0.8)
    return _to_surface(rgb)


def _make_halo():
    r, _ = _grid()
    f = np.clip(1 - r, 0, 1)
    rgb = (f ** 2.5)[..., None] * np.array([255, 80, 210]) * 0.55 + (f ** 1.2)[..., None] * np.array([90, 20, 170]) * 0.6
    return _to_surface(rgb)


def _make_swirl():
    r, a = _grid()
    rng = np.random.default_rng(7)
    arms = 0.5 + 0.5 * np.sin(4 * a + 9 * r + 1.5 * np.sin(3 * a - 5 * r))
    arms2 = 0.5 + 0.5 * np.sin(7 * a - 6 * r)
    f = np.clip(1 - r, 0, 1)
    v = (arms ** 2.2 * 0.8 + arms2 ** 3 * 0.4) * f ** 0.9
    v *= 0.85 + 0.15 * rng.random(v.shape)
    rgb = v[..., None] * np.array([255, 120, 235]) + (v ** 3)[..., None] * np.array([255, 200, 255])
    return _to_surface(rgb)


class SpriteBank:
    """Cached, brightness-quantised glow sprites."""

    def __init__(self):
        self.bases = {"orb": _make_orb(), "halo": _make_halo(), "swirl": _make_swirl()}
        self.cache = {}

    def get(self, kind, w, h, strength):
        w, h = max(4, int(round(w / 8)) * 8), max(4, int(round(h / 8)) * 8)
        q = int(round(min(max(strength, 0.0), 1.6) * 8))
        if q <= 0:
            return None
        key = (kind, w, h, q)
        spr = self.cache.get(key)
        if spr is None:
            if len(self.cache) > 220:
                self.cache.clear()
            spr = pygame.transform.smoothscale(self.bases[kind], (w, h))
            if q < 8:
                m = q * 32
                spr.fill((m, m, m), special_flags=pygame.BLEND_RGB_MULT)
            elif q > 8:
                extra = spr.copy()
                m = min(255, int((q - 8) / 8 * 255))
                extra.fill((m, m, m), special_flags=pygame.BLEND_RGB_MULT)
                spr.blit(extra, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
                spr.blit(extra, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
            self.cache[key] = spr
        return spr

    def draw(self, surf, kind, center, radius, strength=1.0, squash=(1.0, 1.0), angle=0.0):
        if radius < 2:
            return
        spr = self.get(kind, 2 * radius * squash[0], 2 * radius * squash[1], strength)
        if spr is None:
            return
        if angle:
            spr = pygame.transform.rotate(spr, angle)
        surf.blit(spr, spr.get_rect(center=(int(center[0]), int(center[1]))), special_flags=pygame.BLEND_RGB_ADD)


# ---------------------------------------------------------------- particles


class ParticleSystem:
    def __init__(self, cap):
        self.cap = cap
        self.n = 0
        self.pos = np.zeros((cap, 2), np.float32)
        self.vel = np.zeros((cap, 2), np.float32)
        self.life = np.zeros(cap, np.float32)
        self.maxlife = np.ones(cap, np.float32)
        self.bright = np.ones(cap, np.float32)
        self._arrays = (self.pos, self.vel, self.life, self.maxlife, self.bright)

    def set_cap(self, cap):
        if cap == self.cap:
            return
        old = self
        self.__init__(cap)
        k = min(old.n, cap)
        for dst, src in zip(self._arrays, old._arrays):
            dst[:k] = src[:k]
        self.n = k

    def clear(self):
        self.n = 0

    def emit(self, pos, vel, life, bright):
        k = min(len(pos), self.cap - self.n)
        if k <= 0:
            return
        s = slice(self.n, self.n + k)
        self.pos[s], self.vel[s] = pos[:k], vel[:k]
        self.life[s] = self.maxlife[s] = life[:k]
        self.bright[s] = bright[:k]
        self.n += k

    def update(self, dt, t, attractors, drag=1.1):
        n = self.n
        if n == 0:
            return
        p, v = self.pos[:n], self.vel[:n]
        # gravity-free, organic turbulence
        acc = np.empty((n, 2), np.float32)
        acc[:, 0] = 90 * np.sin(p[:, 1] * 0.011 + t * 2.1)
        acc[:, 1] = 90 * np.cos(p[:, 0] * 0.009 + t * 1.7)
        for cx, cy, pull, swirl, rad in attractors:
            d = np.array([cx, cy], np.float32) - p
            dist = np.sqrt((d * d).sum(axis=1)) + 1.0
            fall = (1.0 / (1.0 + dist / rad))[:, None]
            dn = d / dist[:, None]
            acc += dn * pull * fall + np.stack([-dn[:, 1], dn[:, 0]], axis=1) * swirl * fall
        v += acc * dt
        v *= max(0.0, 1.0 - drag * dt)
        p += v * dt
        self.life[:n] -= dt
        keep = np.nonzero(self.life[:n] > 0)[0]
        m = len(keep)
        if m != n:
            for arr in self._arrays:
                arr[:m] = arr[keep]
            self.n = m

    def buffer(self, lw, lh, ds):
        """Accumulate particles into an (lh, lw, 3) float buffer in 0..255 range."""
        n = self.n
        if n == 0:
            return np.zeros((lh, lw, 3), np.float32)
        ix = (self.pos[:n, 0] / ds).astype(np.int32)
        iy = (self.pos[:n, 1] / ds).astype(np.int32)
        ok = (ix >= 0) & (ix < lw) & (iy >= 0) & (iy < lh)
        ix, iy = ix[ok], iy[ok]
        frac = (self.life[:n] / self.maxlife[:n])[ok]
        col = GRADIENT_LUT[np.clip(((1 - frac) * 63).astype(np.int32), 0, 63)]
        inten = (self.bright[:n][ok] * np.minimum(1.0, frac * 2.5))[:, None]
        w = col * inten
        idx = iy * lw + ix
        out = np.stack([np.bincount(idx, weights=w[:, c], minlength=lw * lh) for c in range(3)], axis=1)
        return out.reshape(lh, lw, 3).astype(np.float32)


# ---------------------------------------------------------------- effect objects


class Shockwave:
    def __init__(self, pos, max_r, life=0.8, flash=0.0, strength=1.0):
        self.pos, self.max_r, self.life, self.age = np.array(pos, float), max_r, life, 0.0
        self.flash, self.strength = flash, strength

    def update(self, dt):
        self.age += dt
        return self.age < self.life

    def draw(self, eng, intensity):
        u = self.age / self.life
        r = self.max_r * (1 - (1 - u) ** 2.2)
        s = (1 - u) ** 1.3 * self.strength * min(1.3, intensity)
        if self.flash > 0 and u < 0.35:
            eng._spr("orb", self.pos, self.flash * (0.6 + u), (1 - u / 0.35) * 1.2 * intensity)
        if r < 3 or s < 0.03:
            return
        c = (int(self.pos[0]), int(self.pos[1]))
        for wm, col in LAYERS[:3]:
            w = max(1, int((10 * (1 - u) + 2) * wm / 1.8))
            color = tuple(int(v * min(1.0, s)) for v in col)
            eng._circ(color, c, int(r + w * 0.3), w)
        eng._spr("halo", c, r * 1.15, s * 0.5)


class Projectile:
    def __init__(self, pos, vel, radius, depth):
        self.pos, self.vel = np.array(pos, float), np.array(vel, float)
        self.radius, self.depth, self.age = radius, depth, 0.0
        self.trail = deque(maxlen=22)


class Whip:
    N = 26

    def __init__(self, pts, direction):
        self.pts = pts.astype(np.float64)       # index 0 = head
        self.dir, self.age, self.life = direction, 0.0, 1.1
        self.head_v = np.zeros(2)


class HandFX:
    KEYS = ("palm", "wrist", "t4", "t8", "t12", "t16", "t20")

    def __init__(self):
        self.trails = {k: deque(maxlen=70) for k in self.KEYS}
        self.charge = self.compress = self.beam = self.speedf = 0.0
        self.last_pose = NONE
        self.state = None
        self.seen = 0.0
        self.active = False
        self.phase = np.random.random() * TAU
        self.anchor = None


# ---------------------------------------------------------------- engine


class EnergyEngine:
    def __init__(self, size):
        self.W, self.H = size
        self.bank = SpriteBank()
        self.rng = np.random.default_rng()
        self.quality = config.DEFAULT_QUALITY
        self.ds = config.QUALITY_PRESETS[self.quality][1]
        self.particles = ParticleSystem(config.QUALITY_PRESETS[self.quality][0])
        self.intensity = 1.0
        self.adapt = 1.0
        self._adapt_t = 0.0
        self.hands = {}
        self.projectiles, self.shockwaves, self.whips = [], [], []
        self.attractors = []
        self._lines, self._sprites, self._circles = [], [], []
        self.arcs = []                   # transient lightning polylines for this frame
        self.amt = {RAISED: 0.0, APART: 0.0, TOGETHER: 0.0}
        self.two = None
        self._wave_t = 0.0
        self.t = 0.0
        self.mode_label = NONE

    # -- configuration -------------------------------------------------------
    def resize(self, size):
        self.W, self.H = size

    def set_quality(self, name):
        self.quality = name
        cap, self.ds = config.QUALITY_PRESETS[name]
        self.particles.set_cap(cap)

    def clear(self):
        self.particles.clear()
        self.hands.clear()
        self.projectiles.clear()
        self.shockwaves.clear()
        self.whips.clear()
        self.attractors.clear()
        self.amt = {k: 0.0 for k in self.amt}

    @property
    def particle_count(self):
        return self.particles.n

    def _mult(self):
        return self.intensity * self.adapt * (0.5 + self.particles.cap / 12000.0)

    def _adapt(self, fps, dt):
        self._adapt_t += dt
        if self._adapt_t < 0.5 or fps <= 0:
            return
        self._adapt_t = 0.0
        if fps < config.ADAPTIVE_LOW_FPS:
            self.adapt = max(config.ADAPTIVE_MIN_SCALE, self.adapt * 0.85)
        elif fps > config.ADAPTIVE_HIGH_FPS:
            self.adapt = min(1.0, self.adapt * 1.06)

    # -- emission helpers ----------------------------------------------------
    def _count(self, rate, dt):
        x = rate * dt * self._mult()
        n = int(x)
        return n + (1 if self.rng.random() < x - n else 0)

    def _emit_ring(self, c, radius, n, swirl, inward, life=(0.5, 1.2), bright=(0.6, 1.0)):
        if n <= 0:
            return
        r = self.rng
        a = r.uniform(0, TAU, n)
        rad = np.stack([np.cos(a), np.sin(a)], 1)
        tan = np.stack([-np.sin(a), np.cos(a)], 1)
        pos = np.asarray(c) + rad * (radius * r.uniform(0.6, 1.4, n))[:, None]
        vel = tan * swirl - rad * inward
        self.particles.emit(pos, vel, r.uniform(*life, n), r.uniform(*bright, n))

    def _emit_burst(self, c, n, speed, life=(0.4, 1.1), bright=(0.8, 1.3)):
        n = max(0, int(n * self._mult()))
        if n == 0:
            return
        r = self.rng
        a = r.uniform(0, TAU, n)
        sp = speed * r.uniform(0.15, 1.0, n) ** 0.7
        vel = np.stack([np.cos(a), np.sin(a)], 1) * sp[:, None]
        self.particles.emit(np.tile(np.asarray(c, float), (n, 1)), vel, r.uniform(*life, n), r.uniform(*bright, n))

    def _emit_dir(self, c, d, n, speed, cone=0.25, life=(0.3, 0.8), bright=(0.7, 1.2)):
        if n <= 0:
            return
        r = self.rng
        a = math.atan2(d[1], d[0]) + r.normal(0, cone, n)
        sp = speed * r.uniform(0.3, 1.0, n)
        vel = np.stack([np.cos(a), np.sin(a)], 1) * sp[:, None]
        pos = np.asarray(c, float) + r.normal(0, 3, (n, 2))
        self.particles.emit(pos, vel, r.uniform(*life, n), r.uniform(*bright, n))

    def _emit_line(self, a, b, n, spread=8.0, speed=40.0, life=(0.3, 0.8)):
        if n <= 0:
            return
        r = self.rng
        a, b = np.asarray(a, float), np.asarray(b, float)
        pos = a + (b - a) * r.random((n, 1)) + r.normal(0, spread, (n, 2))
        vel = r.normal(0, speed, (n, 2))
        self.particles.emit(pos, vel, r.uniform(*life, n), r.uniform(0.6, 1.1, n))

    def _shock(self, pos, max_r, life=0.8, flash=0.0, strength=1.0):
        if len(self.shockwaves) < 24:
            self.shockwaves.append(Shockwave(pos, max_r, life, flash, strength))

    # -- main update -----------------------------------------------------------
    def update(self, dt, t, states, two, fps=60.0):
        dt = min(dt, 0.05)
        self.t = t
        self.two = two
        self._adapt(fps, dt)
        self.attractors = []
        self.arcs = []
        H = self.H

        live = set(states)
        for label, st in states.items():
            fx = self.hands.setdefault(label, HandFX())
            self._update_hand(fx, st, t, dt)
        for label, fx in list(self.hands.items()):
            fx.active = label in live
            if label not in live:
                if t - fx.seen > 1.0:
                    del self.hands[label]
                else:
                    fx.charge *= 0.9
                    fx.compress *= 0.9
                    fx.beam = 0.0

        self._update_two_hand(two, dt)
        self._update_projectiles(dt)
        self.shockwaves = [s for s in self.shockwaves if s.update(dt)]
        self._update_whips(dt, t)
        self.particles.update(dt, t, self.attractors)

    def _update_hand(self, fx, st, t, dt):
        H, I = self.H, self.intensity
        fx.state, fx.seen = st, t
        pts = st.pts
        fx.trails["palm"].append((t, st.palm[0], st.palm[1]))
        fx.trails["wrist"].append((t, pts[0][0], pts[0][1]))
        for k, i in (("t4", 4), ("t8", 8), ("t12", 12), ("t16", 16), ("t20", 20)):
            fx.trails[k].append((t, pts[i][0], pts[i][1]))
        target = float(np.clip((st.speed - config.TRAIL_SPEED) / (config.TRAIL_SPEED_MAX - config.TRAIL_SPEED), 0, 1))
        fx.speedf += (target - fx.speedf) * min(1.0, dt * 10)

        pose = st.pose
        tc = min(1.0, st.hold / config.CHARGE_FULL_TIME) if pose == OPEN_PALM else 0.0
        fx.charge += (tc - fx.charge) * min(1.0, dt * (4.0 if tc > fx.charge else 2.5))
        tk = min(1.0, 0.35 + st.hold / 2.5) if pose == FIST else 0.0
        fx.compress += (tk - fx.compress) * min(1.0, dt * 6.0)
        tb = 1.0 if pose == POINT else 0.0
        fx.beam += (tb - fx.beam) * min(1.0, dt * 9.0)

        size = st.size_px
        # Energy floats above the palm (toward the fingers) with a gap, gently bobbing
        bob = math.sin(t * 3.2 + fx.phase) * size * 0.06
        palm = st.palm + st.direction * (size * config.ANCHOR_DISTANCE + bob)
        fx.anchor = palm
        # --- activation bursts
        if pose != fx.last_pose:
            if pose == OPEN_PALM:
                self._emit_burst(palm, 70, 420)
                self._shock(palm, size * 2.2, 0.55, size * 1.4, 0.7)
            elif pose == FIST:
                self._emit_burst(palm, 45, 260, life=(0.2, 0.5))
                self._shock(palm, size * 1.3, 0.35, 0, 0.6)
            elif pose == POINT:
                self._emit_burst(pts[8], 40, 380, life=(0.25, 0.6))
            if fx.last_pose == FIST and pose == OPEN_PALM and fx.compress > 0.5:
                self._emit_burst(palm, 160, 700)
                self._shock(palm, size * 4.0, 0.9, size * 2.0, 1.0)
            fx.last_pose = pose

        # --- ambient aura particles orbiting every hand
        if pose != NONE:
            self._emit_ring(palm, size * 1.5, self._count(35 + 80 * fx.charge, dt), 160, 20)
        self.attractors.append((palm[0], palm[1], 380 + 500 * fx.charge, 300 + 300 * fx.charge, size * 1.8))

        # --- pose behaviour
        if pose == OPEN_PALM:
            ch = fx.charge
            self._emit_ring(palm, size * (1.8 + ch), self._count(90 + 380 * ch, dt), 200 + 150 * ch, 90 + 120 * ch)
            if ch > 0.2 or self.rng.random() < 0.3:
                for _ in range(1 + int(ch * 2)):
                    tip = pts[self.rng.choice([4, 8, 12, 16, 20])]
                    a = lightning_points(palm, tip + (tip - palm) * self.rng.uniform(0.0, 0.5), self.rng)
                    self.arcs.append((a, 3 + 3 * ch, 0.55 + 0.4 * ch))
        elif pose == FIST:
            k = fx.compress
            n = self._count(60 + 220 * k * (1 + st.hold * 0.4), dt)
            if n:
                a = self.rng.uniform(0, TAU, n)
                v = np.stack([np.cos(a), np.sin(a)], 1) * self.rng.uniform(150, 520, n)[:, None]
                self.particles.emit(np.tile(palm, (n, 1)), v, self.rng.uniform(0.2, 0.55, n),
                                    self.rng.uniform(0.9, 1.4, n))
            self.attractors.append((palm[0], palm[1], 900 * k, 400, size))
            if self.rng.random() < 0.25 + 0.4 * k:
                ang = self.rng.uniform(0, TAU)
                end = palm + np.array([math.cos(ang), math.sin(ang)]) * size * self.rng.uniform(0.8, 1.6)
                self.arcs.append((lightning_points(palm, end, self.rng, depth=3), 3, 0.8))
        elif pose == POINT:
            self._update_beam(fx, st, dt)

        # --- fast movement: sparks off fingertips + intense trails
        if fx.speedf > 0.05:
            tips = pts[[4, 8, 12, 16, 20]]
            for tip in tips:
                n = self._count(80 * fx.speedf, dt)
                if n:
                    self._emit_dir(tip, -st.velocity / (np.linalg.norm(st.velocity) + 1e-6), n,
                                   120 + 300 * fx.speedf, cone=0.6, life=(0.3, 0.9))

        # --- events
        for ev in st.events:
            if ev == "thrust":
                self._fire(fx, st)
            elif ev.startswith("sweep"):
                self._whip(fx, st, 1 if ev.endswith("right") else -1)

    def _beam_end(self, st):
        o, d = st.pts[8], st.point_dir
        far = max(self.W, self.H) * 1.6
        return o, o + d * far

    def _update_beam(self, fx, st, dt):
        o, e = self._beam_end(st)
        d = st.point_dir
        n = self._count(260 * fx.beam, dt)
        if n:
            r = self.rng
            s = r.random(n)[:, None] * min(self.W, self.H) * 0.9
            pos = o + d * s + r.normal(0, 4, (n, 2))
            vel = r.normal(0, 25, (n, 2)) + d * 40
            self.particles.emit(pos, vel, r.uniform(0.2, 0.6, n), r.uniform(0.6, 1.1, n))
        self._emit_dir(o, d, self._count(120 * fx.beam, dt), 260, 0.5, (0.15, 0.4), (1.0, 1.4))

    def _fire(self, fx, st):
        H = self.H
        v = st.velocity
        sp = float(np.linalg.norm(v)) / H
        if sp > 0.6:
            d = v / np.linalg.norm(v)
            depth = False
        else:
            d = st.direction
            depth = True       # mostly a toward-the-camera push: projectile flies "into" the scene
        radius = max(st.size_px * 0.6, 0.045 * H) * min(1.5, 0.7 + 0.5 * self.intensity)
        origin = fx.anchor if fx.anchor is not None else st.palm
        self.projectiles.append(Projectile(origin, d * H * 2.4, radius, depth))
        self._emit_burst(origin, 90, 520)
        self._shock(origin, st.size_px * 3.5, 0.6, st.size_px * 1.8, 1.0)

    def _update_projectiles(self, dt):
        W, H = self.W, self.H
        keep = []
        for p in self.projectiles:
            p.age += dt
            p.pos = p.pos + p.vel * dt
            if p.depth:
                p.radius *= max(0.0, 1 - 0.55 * dt)
            p.trail.append(p.pos.copy())
            back = -p.vel / (np.linalg.norm(p.vel) + 1e-6)
            self._emit_dir(p.pos, back, self._count(420, dt), 260, 0.5, (0.3, 0.9), (0.8, 1.3))
            self._emit_ring(p.pos, p.radius * 0.8, self._count(120, dt), 120, 60, (0.2, 0.5))
            m = 60
            out = not (-m < p.pos[0] < W + m and -m < p.pos[1] < H + m)
            if out or p.age > 1.5 or p.radius < 6:
                c = np.clip(p.pos, 20, [W - 20, H - 20])
                self._emit_burst(c, 220, 800, life=(0.4, 1.3))
                self._shock(c, 0.3 * H, 0.9, p.radius * 3.2, 1.0)
                self._shock(c, 0.16 * H, 0.5, 0, 0.8)
            else:
                keep.append(p)
        self.projectiles = keep

    def _whip(self, fx, st, sign):
        N = Whip.N
        tr = [(a, b, c) for a, b, c in fx.trails["palm"] if self.t - a < 0.28]
        if len(tr) >= 2:
            arr = np.array(tr)[::-1]                      # newest first
            u = np.linspace(0, 1, len(arr))
            tt = np.linspace(0, 1, N)
            pts = np.stack([np.interp(tt, u, arr[:, 1]), np.interp(tt, u, arr[:, 2])], 1)
        else:
            pts = np.tile(st.palm, (N, 1))
        if len(self.whips) >= 6:
            self.whips.pop(0)
        w = Whip(pts, sign)
        w.head_v = st.velocity.astype(np.float64) * 0.9
        self.whips.append(w)
        self._emit_burst(st.palm, 40, 350, life=(0.2, 0.6))

    def _update_whips(self, dt, t):
        keep = []
        N = Whip.N
        for w in self.whips:
            w.age += dt
            if w.age >= w.life:
                continue
            if w.age < 0.45 and self.hands:
                # head tracks a live hand if one is nearby, otherwise drifts
                best = min((f.state.palm for f in self.hands.values() if f.state is not None and f.active),
                           key=lambda p: np.linalg.norm(p - w.pts[0]), default=None)
                if best is not None and np.linalg.norm(best - w.pts[0]) < 0.5 * self.H:
                    w.pts[0] += (best - w.pts[0]) * min(1.0, dt * 22)
                else:
                    w.pts[0] += w.head_v * dt
            else:
                w.pts[0] += w.head_v * dt
                w.head_v *= math.exp(-2.5 * dt)
            for i in range(1, N):
                k = 26 * (1 - 0.65 * i / N)
                w.pts[i] += (w.pts[i - 1] - w.pts[i]) * (1 - math.exp(-k * dt))
            if self.rng.random() < 0.8:
                i = self.rng.integers(0, N)
                n = self._count(40, dt)
                self._emit_dir(w.pts[i], self.rng.normal(size=2), n, 90, 3.0, (0.3, 0.8))
            keep.append(w)
        self.whips = keep

    # -- two-hand powers -------------------------------------------------------
    def _update_two_hand(self, two, dt):
        H, W = self.H, self.W
        for k in self.amt:
            tgt = 1.0 if (two.pose == k and len(self.hands) >= 2) else 0.0
            self.amt[k] += (tgt - self.amt[k]) * min(1.0, dt * (5.0 if tgt else 3.0))
        if two.center is None:
            return
        c = two.center
        l = self.hands.get("Left")
        r = self.hands.get("Right")
        if not (l and r and l.state and r.state):
            return
        lp, rp = l.state.palm, r.state.palm
        # activation bursts
        if two.pose != self.mode_label:
            if two.pose == RAISED:
                self._emit_burst(c, 200, 700)
                self._shock(c, 0.5 * H, 1.0, 0.25 * H, 1.0)
            elif two.pose == APART:
                self._emit_burst(c, 120, 600)
            elif two.pose == TOGETHER:
                self._emit_burst(c, 100, 450)
                self._shock(c, 0.2 * H, 0.6, 0.1 * H, 0.8)
            self.mode_label = two.pose

        a = self.amt[RAISED]
        if a > 0.05:
            body = np.array([c[0], min(H * 0.92, c[1] + 0.5 * H)])
            n = self._count(520 * a, dt)
            if n:
                ang = self.rng.uniform(0, TAU, n)
                rx, ry = 0.28 * H, 0.55 * H
                rr = self.rng.uniform(0.5, 1.1, n)
                pos = body + np.stack([np.cos(ang) * rx * rr, np.sin(ang) * ry * rr], 1)
                vel = np.stack([-np.sin(ang), np.cos(ang)], 1) * 180 + np.array([0, -140])
                self.particles.emit(pos, vel, self.rng.uniform(0.7, 1.6, n), self.rng.uniform(0.6, 1.1, n))
            self.attractors.append((body[0], body[1], 300 * a, 520 * a, 0.4 * H))
            self._wave_t -= dt
            if self._wave_t <= 0:
                self._wave_t = 0.55
                self._shock(body, 0.6 * H, 1.1, 0, 0.9 * a)
                self._shock(lp, 0.2 * H, 0.7, 0, 0.8 * a)
                self._shock(rp, 0.2 * H, 0.7, 0, 0.8 * a)
            for p in (lp, rp):
                if self.rng.random() < 0.7:
                    end = body + self.rng.normal(0, 0.08 * H, 2)
                    self.arcs.append((lightning_points(p, end, self.rng), 3 + 2 * a, 0.7 * a))

        a = self.amt[APART]
        if a > 0.05:
            self._emit_line(lp, rp, self._count(260 * a, dt), 0.08 * two.distance_px)
            self.attractors.append((c[0], c[1], 150 * a, 380 * a, 0.3 * H))
            for _ in range(3):
                if self.rng.random() < 0.8:
                    self.arcs.append((lightning_points(lp, rp, self.rng, 0.12), 3, 0.7 * a))

        a = self.amt[TOGETHER]
        if a > 0.05:
            rad, _ = self._orb_params(two)
            self._emit_ring(c, rad * 1.5, self._count(280 * a, dt), 220, 160, (0.4, 1.0))
            self.attractors.append((c[0], c[1], 900 * a, 700 * a, rad * 2))
            for p in (lp, rp):
                if self.rng.random() < 0.6:
                    self.arcs.append((lightning_points(p, c, self.rng, 0.2), 3, 0.8 * a))

    def _orb_params(self, two):
        close = float(np.clip(1 - two.distance / config.TOGETHER_DIST, 0, 1))
        hold = min(1.0, two.hold / 3.0)
        rad = self.H * (0.05 + 0.13 * close) * (1 + 0.6 * hold) * min(1.4, 0.6 + 0.5 * self.intensity)
        return rad, 0.8 + 0.5 * hold

    # -- rendering ---------------------------------------------------------------
    def particle_buffer(self, lw, lh):
        return self.particles.buffer(lw, lh, self.ds)

    def _spr(self, kind, center, radius, strength=1.0, squash=(1.0, 1.0), angle=0.0):
        self._sprites.append((kind, center, radius, strength, squash, angle))

    def _line(self, pts, width, strength=1.0, taper=True, chunks=5):
        segs = line_segments(pts, taper, chunks)
        if segs and strength > 0.01:
            self._lines.append((segs, width, strength))

    def _circ(self, col, center, radius, width):
        self._circles.append((col, center, radius, width))

    def draw(self, fx):
        bank, t, I, H = self.bank, self.t, self.intensity, self.H
        self._lines, self._sprites, self._circles = [], [], []
        any_hand = [f for f in self.hands.values() if f.state is not None and f.active]

        # body / upper-body aura
        if any_hand:
            xs = [f.state.palm[0] for f in any_hand]
            ch = max(f.charge for f in any_hand)
            raised = self.amt[RAISED]
            s = min(1.4, (0.14 + 0.22 * ch + 0.9 * raised) * I)
            bx = sum(xs) / len(xs)
            by = H * 0.9
            if raised > 0.05 and self.two is not None and self.two.center is not None:
                bx, by = self.two.center[0], min(H * 0.95, self.two.center[1] + 0.5 * H)
            pulse = 1 + 0.06 * math.sin(t * 3.0)
            self._spr("halo", (bx, by), 0.55 * H * pulse * (1 + raised * 0.5), s, squash=(0.8, 1.3))
            if raised > 0.05:
                self._spr("swirl", (bx, by), 0.5 * H, 0.5 * raised * I, squash=(0.9, 1.4))

        # trails / ribbons
        for f in self.hands.values():
            self._draw_trails(fx, f)

        # two-hand fields
        self._draw_two(fx)

        # per-hand orbs and beams
        for f in any_hand:
            self._draw_hand(fx, f)

        for w in self.whips:
            self._draw_whip(fx, w)
        for p in self.projectiles:
            self._draw_projectile(fx, p)
        for s in self.shockwaves:
            s.draw(self, I)
        for pts, width, strength in self.arcs:
            self._line(pts, width * min(1.5, 0.7 + 0.4 * I), strength, taper=False)

        # vector lines first, then additive sprites on top so glows are never overwritten
        flush_lines(fx, self._lines)
        for col, c, r, w in self._circles:
            pygame.draw.circle(fx, col, c, r, w)
        for kind, center, radius, strength, squash, angle in self._sprites:
            bank.draw(fx, kind, center, radius, strength, squash, angle)

    def _trail_pts(self, trail, life):
        t = self.t
        arr = [(x, y) for ts, x, y in trail if t - ts < life]
        return np.array(arr, float) if len(arr) >= 2 else None

    def _draw_trails(self, fx, f):
        I = self.intensity
        sp = f.speedf
        life = 0.18 + 0.6 * sp
        fade = 1.0 if f.active else max(0.0, 1 - (self.t - f.seen) / 1.0)
        strength = (0.3 + 0.7 * sp) * fade * min(1.3, I)
        t = self.t
        for key in f.KEYS:
            thin = key not in ("palm", "wrist")
            pts = self._trail_pts(f.trails[key], life * (0.8 if thin else 1.0))
            if pts is None:
                continue
            if key == "wrist" or key == "palm":
                # flowing ribbon: lateral wave that grows toward the tail
                d = np.diff(pts, axis=0, prepend=pts[:1])
                L = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
                perp = np.stack([-d[:, 1], d[:, 0]], 1) / L
                u = np.linspace(1, 0, len(pts))[:, None]
                wave = np.sin(np.linspace(0, 7, len(pts))[:, None] - t * 9 + f.phase) * u * 10 * (0.4 + sp)
                pts = pts + perp * wave
            pts = smooth_path(pts, 3)
            width = (9 if key == "palm" else 6 if key == "wrist" else 3.5) * (0.6 + 0.8 * sp) * min(1.4, 0.7 + 0.3 * I)
            self._line(pts, width, strength * (1.0 if not thin else 0.85))

    def _draw_hand(self, fx, f):
        st, bank, t, I = f.state, self.bank, self.t, self.intensity
        palm, size, pose = f.anchor if f.anchor is not None else st.palm, st.size_px, st.pose
        rot = (t * 70 + f.phase * 50) % 360
        if pose == OPEN_PALM:
            ch = f.charge
            r = size * (0.55 + 0.5 * ch) * (1 + 0.05 * math.sin(t * 7)) * min(1.5, 0.7 + 0.4 * I)
            self._spr("halo", palm, r * 3.0, (0.55 + 0.5 * ch) * I)
            self._spr("swirl", palm, r * 1.5, (0.55 + 0.4 * ch) * I, angle=rot)
            self._spr("swirl", palm, r * 1.1, (0.5 + 0.3 * ch) * I, angle=-rot * 1.6)
            self._spr("orb", palm, r, (0.8 + 0.5 * ch) * min(1.3, I))
        elif pose == FIST:
            k = f.compress
            freq = 2.5 + 7 * k * (1 + st.hold * 0.2)
            pulse = 1 + (0.08 + 0.2 * k) * math.sin(t * freq * TAU / 2)
            r = size * (0.42 - 0.12 * k) * pulse * min(1.5, 0.7 + 0.4 * I)
            self._spr("halo", palm, r * 3.2, (0.5 + 0.5 * k) * I)
            self._spr("orb", palm, r * 1.5, 1.0 + 0.3 * k)
            self._spr("orb", palm, r, 1.6)
        elif pose == POINT:
            tip = st.pts[8]
            self._draw_beam(fx, f, st)
            self._spr("halo", tip, size * 0.9, 0.8 * I)
            self._spr("orb", tip, size * 0.28, 1.2)

    def _draw_beam(self, fx, f, st):
        o, e = self._beam_end(st)
        k = f.beam
        if k < 0.05:
            return
        t = self.t
        d = st.point_dir
        perp = np.array([-d[1], d[0]])
        L = np.linalg.norm(e - o)
        s = np.linspace(0, 1, 18)[:, None]
        jitter = (np.sin(s * 40 - t * 50) * 2.5 + self.rng.normal(0, 0.8, s.shape)) * (0.3 + s)
        pts = o + (e - o) * s + perp * jitter
        self._line(pts, 6 * (0.6 + 0.4 * k) * min(1.5, 0.7 + 0.4 * self.intensity), k, taper=False)
        if self.rng.random() < 0.5:
            a = o + d * L * self.rng.uniform(0.05, 0.4)
            self.arcs.append((lightning_points(a, a + perp * self.rng.uniform(-40, 40) + d * 30, self.rng, depth=3), 2, 0.7 * k))
        # impact spark where the beam leaves the screen
        if 0 <= o[0] <= self.W:
            tt = [(-o[0] / d[0]) if d[0] < -1e-6 else (self.W - o[0]) / d[0] if d[0] > 1e-6 else 1e9,
                  (-o[1] / d[1]) if d[1] < -1e-6 else (self.H - o[1]) / d[1] if d[1] > 1e-6 else 1e9]
            hit = o + d * min(tt)
            self._spr("orb", hit, 28 * k, 0.9)
            self._emit_dir(hit, -d, self._count(60, 0.016), 200, 1.2, (0.2, 0.5))

    def _draw_two(self, fx):
        two, bank, t, I = self.two, self.bank, self.t, self.intensity
        if two is None or two.center is None:
            return
        l, r = self.hands.get("Left"), self.hands.get("Right")
        if not (l and r and l.state and r.state):
            return
        lp, rp, c = l.state.palm, r.state.palm, two.center
        a = self.amt[APART]
        if a > 0.05:
            rad = max(30.0, two.distance_px * 0.5 * 0.95)
            self._spr("halo", c, rad * 1.1, 0.7 * a * I)
            self._spr("swirl", c, rad * 0.9, 0.35 * a * I, angle=(t * 40) % 360)
            for k in range(3):
                u = (t * 0.6 + k / 3) % 1.0
                rr = int(rad * (0.25 + 0.75 * u))
                col = tuple(int(v * (1 - u) * a) for v in (255, 90, 215))
                self._circ(col, (int(c[0]), int(c[1])), rr, 3)
            d = rp - lp
            perp = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-6)
            s = np.linspace(0, 1, 28)[:, None]
            for k, ph in enumerate((0.0, 2.1, 4.2)):
                bow = np.sin(s * math.pi) * np.sin(t * 3 + ph) * 0.12 * two.distance_px
                pts = lp + d * s + perp * bow
                self._line(pts, 4, 0.8 * a, taper=False)
            self._spr("orb", lp, l.state.size_px * 0.6, 0.9 * a)
            self._spr("orb", rp, r.state.size_px * 0.6, 0.9 * a)
        a = self.amt[TOGETHER]
        if a > 0.05:
            rad, br = self._orb_params(two)
            pulse = 1 + 0.07 * math.sin(t * 9)
            rad *= a * pulse
            rot = (t * 90) % 360
            self._spr("halo", c, rad * 3.0, 0.6 * br * I)
            self._spr("swirl", c, rad * 1.6, 0.5 * br * I, angle=rot)
            self._spr("swirl", c, rad * 1.2, 0.4 * br * I, angle=-rot * 1.5)
            self._spr("orb", c, rad, 0.75 * br)
        a = self.amt[RAISED]
        if a > 0.05:
            for f in (l, r):
                p = f.state.palm
                self._spr("halo", p, f.state.size_px * 3.2, 0.8 * a * I)
                self._spr("swirl", p, f.state.size_px * 1.7, 0.5 * a * I, angle=(t * 120) % 360)

    def _draw_whip(self, fx, w):
        u = w.age / w.life
        s = (1 - u) ** 0.8 * min(1.3, self.intensity)
        N = Whip.N
        pts = w.pts.copy()
        d = np.diff(pts, axis=0, prepend=pts[:1] - 1e-3)
        L = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
        perp = np.stack([-d[:, 1], d[:, 0]], 1) / L
        idx = np.arange(N)[:, None]
        wave = np.sin(idx * 0.7 - self.t * 14) * (idx / N) * 14 * (1 - u)
        pts = pts + perp * wave
        pts = smooth_path(pts[::-1], 3)         # tail -> head so the head is widest
        self._line(pts, 15 * min(1.5, 0.7 + 0.4 * self.intensity), s, chunks=7)

    def _draw_projectile(self, fx, p):
        bank = self.bank
        if len(p.trail) >= 2:
            pts = smooth_path(np.array(p.trail), 2)
            self._line(pts, p.radius * 0.9, 0.9, chunks=6)
        r = p.radius
        self._spr("halo", p.pos, r * 3.2, 0.9)
        self._spr("swirl", p.pos, r * 1.6, 0.8, angle=(self.t * 400) % 360)
        self._spr("orb", p.pos, r * 1.15, 1.4)
