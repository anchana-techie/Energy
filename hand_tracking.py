"""MediaPipe hand landmark integration with temporal smoothing.

Works with both the classic ``mp.solutions.hands`` API and the newer Tasks
``HandLandmarker`` API (the model file is downloaded once if that is needed).
All processing is local; frames are never stored or uploaded.
"""
import os
import time
import urllib.request
from dataclasses import dataclass

import numpy as np

import config


@dataclass
class Hand:
    label: str            # "Left" or "Right" (as the user sees their own hands)
    lm: np.ndarray        # (21, 3) normalized x, y in the mirrored frame, relative z
    score: float = 1.0


class HandTracker:
    def __init__(self):
        import mediapipe as mp  # imported lazily so main can report a missing package
        self._mp = mp
        self._solutions = None
        self._landmarker = None
        self._ts_ms = 0
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self._solutions = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=config.MAX_HANDS,
                model_complexity=1,
                min_detection_confidence=config.DETECTION_CONFIDENCE,
                min_tracking_confidence=config.TRACKING_CONFIDENCE,
            )
        else:
            self._landmarker = self._make_landmarker()
        self._smooth = {}    # label -> (landmarks, last_seen_time)

    # -- setup ---------------------------------------------------------------
    def _make_landmarker(self):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.MODEL_FILE)
        if not os.path.exists(path):
            try:
                urllib.request.urlretrieve(config.MODEL_URL, path)
            except Exception as exc:
                raise RuntimeError(
                    "This MediaPipe build needs the hand_landmarker.task model file.\n"
                    f"Download it from {config.MODEL_URL}\nand place it next to main.py. ({exc})")
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config.MAX_HANDS,
            min_hand_detection_confidence=config.DETECTION_CONFIDENCE,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
        )
        return vision.HandLandmarker.create_from_options(options)

    # -- detection -----------------------------------------------------------
    def _detect_raw(self, rgb):
        """Return list of (label, (21,3) array, score) from an already-mirrored RGB frame."""
        out = []
        if self._solutions is not None:
            rgb.flags.writeable = False
            res = self._solutions.process(rgb)
            if res.multi_hand_landmarks:
                for lms, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    arr = np.array([[p.x, p.y, p.z] for p in lms.landmark], dtype=np.float32)
                    cls = hd.classification[0]
                    out.append((cls.label, arr, cls.score))
        else:
            self._ts_ms = max(self._ts_ms + 1, int(time.monotonic() * 1000))
            img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            res = self._landmarker.detect_for_video(img, self._ts_ms)
            for lms, hd in zip(res.hand_landmarks, res.handedness):
                arr = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
                out.append((hd[0].category_name, arr, hd[0].score))
        return out

    def process(self, rgb_mirrored):
        """Detect hands in a mirrored RGB frame and return smoothed Hand objects."""
        now = time.monotonic()
        raw = self._detect_raw(np.ascontiguousarray(rgb_mirrored))
        # Because the frame is already mirrored, MediaPipe's label matches the user's real hand.
        # With two hands, positions are more reliable than labels (which can duplicate).
        if len(raw) == 2:
            raw.sort(key=lambda r: float(r[1][0, 0]))
            raw = [("Left", raw[0][1], raw[0][2]), ("Right", raw[1][1], raw[1][2])]
        for label, arr, score in raw:
            prev = self._smooth.get(label)
            if prev is not None and now - prev[1] < config.HAND_LOST_GRACE * 2:
                a = config.LANDMARK_SMOOTHING
                arr = prev[0] * (1 - a) + arr * a
            self._smooth[label] = (arr, now, score)
        hands = []
        for label, (arr, seen, score) in list(self._smooth.items()):
            if now - seen > config.HAND_LOST_GRACE:
                del self._smooth[label]       # graceful recovery: forget stale hands
                continue
            hands.append(Hand(label, arr, score))
        hands.sort(key=lambda h: h.label)
        return hands

    def close(self):
        for obj in (self._solutions, self._landmarker):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        self._solutions = self._landmarker = None
