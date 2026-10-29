"""Webcam capture + hand tracking on a background thread.

The worker keeps only the most recent frame in memory; nothing is written to disk.
"""
import threading
import time

import cv2

import config
from hand_tracking import HandTracker


class CameraWorker:
    def __init__(self):
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame = None        # mirrored RGB uint8 (h, w, 3)
        self._hands = []
        self._seq = 0
        self.error = None
        self.camera_ok = False
        self.tracking_ok = False
        self.running = False
        self.capture_fps = 0.0

    # -- lifecycle -----------------------------------------------------------
    def start(self):
        self.stop()
        self._stop.clear()
        self.error = None
        self.camera_ok = self.tracking_ok = False
        self.running = True
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._thread = None
        self.running = False
        self.camera_ok = self.tracking_ok = False
        with self._lock:
            self._frame, self._hands = None, []

    def latest(self):
        """Return (seq, rgb_frame_or_None, hands)."""
        with self._lock:
            return self._seq, self._frame, list(self._hands)

    # -- worker --------------------------------------------------------------
    def _open_camera(self):
        cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self):
        cap = None
        tracker = None
        try:
            try:
                tracker = HandTracker()
            except ImportError:
                self.error = "MediaPipe is not installed. Run: pip install -r requirements.txt"
                return
            except Exception as exc:
                self.error = f"Could not start hand tracking: {exc}"
                return

            cap = self._open_camera()
            if cap is None:
                self.error = ("No webcam found, or access was denied.\n"
                              "Check that a camera is connected, that no other app is using it, and that "
                              "Settings > Privacy & security > Camera allows desktop apps.")
                return

            fails, last_t, fps = 0, time.monotonic(), 0.0
            while not self._stop.is_set():
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    fails += 1
                    self.camera_ok = False
                    if fails > 60:
                        self.error = "The camera stopped delivering frames (unplugged or blocked?)."
                        return
                    time.sleep(0.03)
                    continue
                fails = 0
                self.camera_ok = True
                bgr = cv2.flip(bgr, 1)                              # mirror horizontally
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                small = cv2.resize(rgb, (config.TRACK_WIDTH, int(h * config.TRACK_WIDTH / w)),
                                   interpolation=cv2.INTER_AREA)
                try:
                    hands = tracker.process(small)
                except Exception as exc:
                    self.error = f"Hand tracking failed: {exc}"
                    return
                self.tracking_ok = bool(hands)
                now = time.monotonic()
                fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-3)
                last_t = now
                self.capture_fps = fps
                with self._lock:
                    self._frame, self._hands = rgb, hands
                    self._seq += 1
        except Exception as exc:  # never let the thread die silently
            self.error = f"Camera error: {exc}"
        finally:
            if cap is not None:
                cap.release()
            if tracker is not None:
                tracker.close()
            self.camera_ok = self.tracking_ok = False
