"""Gesture classification and movement analysis.

Input: hands with landmarks in *screen pixels*. All distances/velocities are normalised by the
screen height so thresholds do not depend on window size.

Per hand it produces a static pose (OPEN_PALM / FIST / POINT / NONE) with temporal smoothing and
hysteresis, plus motion events (thrust, sweep, fast-move). Across both hands it produces a
two-hand pose (RAISED / APART / TOGETHER).
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import config

OPEN_PALM, FIST, POINT, NONE = "OPEN_PALM", "FIST", "POINT", "NONE"
RAISED, APART, TOGETHER = "BOTH_RAISED", "HANDS_APART", "HANDS_TOGETHER"

WRIST = 0
FINGERS = {"index": (5, 6, 8), "middle": (9, 10, 12), "ring": (13, 14, 16), "pinky": (17, 18, 20)}


@dataclass
class HandState:
    label: str
    pts: np.ndarray                  # (21, 2) screen px
    pose: str = NONE
    confidence: float = 0.0
    hold: float = 0.0                # seconds the current pose has been held
    palm: np.ndarray = None          # palm centre, px
    size_px: float = 1.0             # apparent palm size, px
    direction: np.ndarray = None     # unit vector wrist -> fingers
    point_dir: np.ndarray = None     # unit vector of index finger
    facing: float = 0.0              # >0 palm toward camera, <0 back of hand
    velocity: np.ndarray = None      # px/s
    speed: float = 0.0               # H/s
    accel: float = 0.0               # H/s^2 (magnitude)
    growth: float = 0.0              # d ln(size)/dt, forward thrust indicator
    events: list = field(default_factory=list)   # "thrust", "sweep_left", "sweep_right"
    fingers_ext: dict = field(default_factory=dict)
    facing_ok: bool = False          # palm side toward the camera (back of hand = no magic)


@dataclass
class TwoHandState:
    pose: str = NONE
    hold: float = 0.0
    distance_px: float = 0.0
    distance: float = 0.0            # in H units
    center: np.ndarray = None
    closing_speed: float = 0.0       # H/s, positive when hands approach


class _Tracker:
    """Per-hand temporal state."""

    def __init__(self):
        self.hist = deque(maxlen=40)         # (t, palm_h, ln_size)
        self.scores = {OPEN_PALM: 0.0, FIST: 0.0, POINT: 0.0, NONE: 0.3}
        self.pose = NONE
        self.pose_since = 0.0
        self.challenger, self.challenger_n = NONE, 0
        self.last_thrust = -9.0
        self.last_sweep = -9.0
        self.prev_speed = 0.0
        self.last_seen = 0.0
        self.speed_s = 0.0
        self.facing = 0.0
        self.facing_ok = False


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.array([0.0, -1.0])


class GestureEngine:
    def __init__(self):
        self._t = {}                 # label -> _Tracker
        self._two_pose = NONE
        self._two_since = 0.0
        self._two_cand, self._two_cand_since = NONE, 0.0
        self._prev_dist, self._prev_dist_t = None, 0.0
        self._closing = 0.0
        self.thrust_scale = 1.0      # user-calibratable sensitivity (higher = easier)

    def reset(self):
        self._t.clear()
        self._two_pose = NONE
        self._prev_dist = None

    def adjust_thrust(self, delta):
        self.thrust_scale = float(np.clip(self.thrust_scale + delta, 0.4, 2.5))
        return self.thrust_scale

    # ------------------------------------------------------------------ main
    def update(self, hands, t, screen_h):
        """hands: list of (label, pts_px (21,2), z (21,)). Returns (dict label->HandState, TwoHandState)."""
        states = {}
        for label, pts, z in hands:
            tr = self._t.setdefault(label, _Tracker())
            states[label] = self._analyse(label, pts, z, tr, t, screen_h)
            tr.last_seen = t
        for label in [k for k, v in self._t.items() if t - v.last_seen > 0.6]:
            del self._t[label]       # forget hands that left the frame
        states = {k: v for k, v in states.items() if v.facing_ok}   # back of hand: no effects
        two = self._two_hand(states, t, screen_h)
        return states, two

    # ------------------------------------------------------- single hand
    def _analyse(self, label, pts, z, tr, t, H):
        st = HandState(label=label, pts=pts)
        wrist = pts[WRIST]
        mid_mcp = pts[9]
        palm_pts = pts[[0, 5, 9, 13, 17]]
        st.palm = palm_pts.mean(axis=0)
        # Apparent size: wrist->middle MCP length blended with knuckle-row width (rotation tolerant)
        st.size_px = max(1.0, 0.5 * np.linalg.norm(mid_mcp - wrist)
                         + 0.5 * np.linalg.norm(pts[5] - pts[17]) * 1.15)
        st.direction = _unit(mid_mcp - wrist)
        st.point_dir = _unit(pts[8] - pts[5])

        # Finger extension scores (tip distance vs knuckle distance from wrist)
        ext = {}
        for name, (mcp, pip, tip) in FINGERS.items():
            r = np.linalg.norm(pts[tip] - wrist) / max(np.linalg.norm(pts[mcp] - wrist), 1e-6)
            ext[name] = float(np.clip((r - 1.25) / 0.55, 0.0, 1.0))
        st.fingers_ext = ext
        others = (ext["middle"] + ext["ring"] + ext["pinky"]) / 3.0
        mean4 = sum(ext.values()) / 4.0
        raw = {
            OPEN_PALM: mean4 * (0.55 + 0.45 * min(ext.values()) / max(mean4, 1e-6)),
            FIST: (1.0 - mean4) * (1.0 if max(ext.values()) < 0.45 else 0.4),
            POINT: ext["index"] * (1.0 - others),
        }
        # Palm orientation from the landmark normal (z in MediaPipe is roughly in image-x units)
        v1, v2 = pts[5] - wrist, pts[17] - wrist
        cross_z = v1[0] * v2[1] - v1[1] * v2[0]
        st.facing = float(np.clip(cross_z / (st.size_px ** 2) * (1 if label == "Right" else -1) * 2.0, -1, 1))

        tr.facing += (st.facing - tr.facing) * 0.4
        tr.facing_ok = tr.facing > (config.FACING_LEAVE if tr.facing_ok else config.FACING_ENTER)
        st.facing, st.facing_ok = tr.facing, tr.facing_ok

        # Temporal smoothing + hysteresis
        a = config.GESTURE_SCORE_EMA
        for k, v in raw.items():
            tr.scores[k] = tr.scores[k] * (1 - a) + v * a
        best = max((OPEN_PALM, FIST, POINT), key=lambda k: tr.scores[k])
        best_score = tr.scores[best]
        cur_score = tr.scores.get(tr.pose, 0.0) if tr.pose != NONE else 0.0
        if tr.pose != NONE and cur_score >= config.GESTURE_KEEP and best == tr.pose:
            tr.challenger, tr.challenger_n = tr.pose, 0
        else:
            candidate = best if best_score >= config.GESTURE_ENTER else (
                tr.pose if (tr.pose != NONE and cur_score >= config.GESTURE_KEEP) else NONE)
            if candidate == tr.pose:
                tr.challenger, tr.challenger_n = tr.pose, 0
            elif candidate == tr.challenger:
                tr.challenger_n += 1
                if tr.challenger_n >= config.GESTURE_MIN_FRAMES:
                    tr.pose, tr.pose_since = candidate, t
                    tr.challenger_n = 0
            else:
                tr.challenger, tr.challenger_n = candidate, 1
        st.pose = tr.pose
        st.confidence = tr.scores.get(tr.pose, 0.0) if tr.pose != NONE else 0.0
        st.hold = t - tr.pose_since if tr.pose != NONE else 0.0

        # Motion history (units of screen height)
        tr.hist.append((t, st.palm / H, float(np.log(st.size_px))))
        self._motion(st, tr, t, H)
        return st

    def _motion(self, st, tr, t, H):
        hist = tr.hist
        st.velocity = np.zeros(2)
        if len(hist) < 3:
            return
        # Use the oldest sample inside the velocity window
        t1, p1, s1 = hist[-1]
        t0, p0, s0 = hist[-2]
        for item in reversed(hist):
            if t1 - item[0] >= config.VELOCITY_WINDOW:
                t0, p0, s0 = item
                break
        else:
            t0, p0, s0 = hist[0]
        dt = max(t1 - t0, 1e-3)
        vel_h = (p1 - p0) / dt
        st.velocity = vel_h * H
        tr.speed_s = tr.speed_s * 0.5 + float(np.linalg.norm(vel_h)) * 0.5
        st.speed = tr.speed_s
        st.growth = (s1 - s0) / dt
        st.accel = (st.speed - tr.prev_speed) / max(t1 - hist[-2][0], 1e-3)
        tr.prev_speed = st.speed

        if len(hist) < 5:
            return
        lateral = abs(vel_h[0])
        # Forward thrust: apparent size grows quickly while the hand does not slide sideways
        thr = config.THRUST_GROWTH_RATE / self.thrust_scale
        if (st.growth > thr and lateral < config.THRUST_MAX_LATERAL
                and st.pose != POINT and t - tr.last_thrust > config.THRUST_COOLDOWN):
            st.events.append("thrust")
            tr.last_thrust = t
        # Sideways sweep
        if (lateral > config.SWEEP_SPEED and lateral > abs(vel_h[1]) * config.SWEEP_AXIS_RATIO
                and abs(st.growth) < thr * 0.8 and t - tr.last_sweep > config.SWEEP_COOLDOWN):
            st.events.append("sweep_right" if vel_h[0] > 0 else "sweep_left")
            tr.last_sweep = t

    # --------------------------------------------------------- two hands
    def _two_hand(self, states, t, H):
        two = TwoHandState()
        if len(states) < 2:
            self._two_pose, self._prev_dist = NONE, None
            self._two_cand = NONE
            return two
        l, r = states.get("Left"), states.get("Right")
        if l is None or r is None:
            return two
        d_px = float(np.linalg.norm(l.palm - r.palm))
        d = d_px / H
        two.distance_px, two.distance = d_px, d
        two.center = (l.palm + r.palm) / 2.0

        if self._prev_dist is not None and t > self._prev_dist_t:
            inst = (self._prev_dist - d) / (t - self._prev_dist_t)
            self._closing = self._closing * 0.7 + inst * 0.3
        self._prev_dist, self._prev_dist_t = d, t
        two.closing_speed = self._closing

        raised = (l.pts[WRIST][1] / H < config.RAISED_Y and r.pts[WRIST][1] / H < config.RAISED_Y
                  and l.pose == OPEN_PALM and r.pose == OPEN_PALM)
        if raised:
            cand = RAISED
        elif d < config.TOGETHER_DIST:
            cand = TOGETHER
        elif d > config.APART_DIST and l.pose != NONE and r.pose != NONE:
            cand = APART
        else:
            cand = NONE
        if cand != self._two_cand:
            self._two_cand, self._two_cand_since = cand, t
        if cand != self._two_pose and t - self._two_cand_since >= config.TWO_HAND_HOLD:
            self._two_pose, self._two_since = cand, t
        two.pose = self._two_pose
        two.hold = t - self._two_since if self._two_pose != NONE else 0.0
        return two
