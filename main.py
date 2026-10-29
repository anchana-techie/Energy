"""Gwen Tennyson Anodite Energy Simulator - entry point.

Run:  python main.py
"""
import sys
import time


def _check_dependencies():
    missing = []
    for mod, pkg in (("numpy", "numpy"), ("cv2", "opencv-python"), ("pygame", "pygame"),
                     ("mediapipe", "mediapipe")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("Missing Python packages: " + ", ".join(missing))
        print("Install them inside your virtual environment with:\n    pip install -r requirements.txt")
        sys.exit(1)


_check_dependencies()

import numpy as np  # noqa: E402
import pygame  # noqa: E402

import config  # noqa: E402
from camera import CameraWorker  # noqa: E402
from energy_effects import EnergyEngine  # noqa: E402
from gesture_recognition import (APART, FIST, NONE, OPEN_PALM, POINT, RAISED,  # noqa: E402
                                 TOGETHER, GestureEngine)
from renderer import Renderer  # noqa: E402
from ui import UI  # noqa: E402

POSE_NAMES = {OPEN_PALM: "Open palm - Charge", FIST: "Fist - Compress", POINT: "Point - Precision beam",
              NONE: "-"}
TWO_NAMES = {RAISED: "Both raised - MAXIMUM POWER", APART: "Hands apart - Expansion",
             TOGETHER: "Hands together - Energy orb", NONE: "-"}
EVENT_NAMES = {"thrust": "ENERGY BLAST!", "sweep_left": "ENERGY WHIP!", "sweep_right": "ENERGY WHIP!"}


class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(config.APP_TITLE)
        self.fullscreen = False
        self.windowed_size = config.WINDOW_SIZE
        self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)
        size = self.screen.get_size()
        self.renderer = Renderer(size)
        self.engine = EnergyEngine(size)
        self.gestures = GestureEngine()
        self.ui = UI(size)
        self.camera = CameraWorker()
        self.clock = pygame.time.Clock()
        self.fps = 0.0
        self.hand_mode = "Both"
        self.last_seq = -1
        self.last_data_t = 0.0
        self.states, self.two = {}, None
        self.rgb = None
        self.error = None
        self.flash, self.flash_t = "", 0.0
        self.manual_blast = False
        self.running = True

    # -- helpers ---------------------------------------------------------------
    def set_fullscreen(self, on):
        self.fullscreen = on
        if on:
            self.windowed_size = self.screen.get_size()
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.ui.show_panel = False      # cinematic mode
        else:
            self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)
            self.ui.show_panel = True
        self._resized()

    def _resized(self):
        size = self.screen.get_size()
        self.renderer.resize(size)
        self.engine.resize(size)
        self.ui.layout(size)

    def start_camera(self):
        self.error = None
        self.gestures.reset()
        self.engine.clear()
        self.camera.start()

    def stop_camera(self):
        self.camera.stop()
        self.engine.clear()
        self.gestures.reset()
        self.states, self.two = {}, None
        self.rgb = None

    def do(self, action):
        if action == "start":
            self.start_camera()
        elif action == "stop":
            self.stop_camera()
        elif action == "reset":
            self.engine.clear()
            self.gestures.reset()
        elif action == "fullscreen":
            self.set_fullscreen(not self.fullscreen)
        elif action == "guide":
            self.ui.show_guide = not self.ui.show_guide
        elif isinstance(action, tuple):
            kind, val = action
            if kind == "intensity":
                self.engine.intensity = val
            elif kind == "quality":
                self.ui.quality = val
                self.engine.set_quality(val)
            elif kind == "hands":
                self.ui.hand_mode = self.hand_mode = val

    def on_key(self, e):
        k = e.key
        if k == pygame.K_ESCAPE:
            if self.fullscreen:
                self.set_fullscreen(False)
            else:
                self.running = False
        elif k == pygame.K_q:
            self.running = False
        elif k in (pygame.K_F11, pygame.K_f):
            self.set_fullscreen(not self.fullscreen)
        elif k == pygame.K_RETURN and not self.camera.running:
            self.start_camera()
        elif k == pygame.K_r:
            self.do("reset")
        elif k == pygame.K_h:
            self.ui.show_panel = not self.ui.show_panel
        elif k == pygame.K_g:
            self.ui.show_guide = not self.ui.show_guide
        elif k == pygame.K_SPACE:
            self.manual_blast = True
        elif k == pygame.K_LEFTBRACKET:
            self.gestures.adjust_thrust(-config.THRUST_SENSITIVITY_STEP)
        elif k == pygame.K_RIGHTBRACKET:
            self.gestures.adjust_thrust(config.THRUST_SENSITIVITY_STEP)

    # -- per-frame ---------------------------------------------------------------
    def poll_camera(self, now):
        if not self.camera.running:
            return
        if self.camera.error:
            self.error = self.camera.error
            self.stop_camera()
            return
        seq, rgb, hands = self.camera.latest()
        if rgb is None:
            return
        self.rgb = rgb
        if seq == self.last_seq:
            return
        self.last_seq = seq
        self.last_data_t = now
        self.renderer.frame_size = (rgb.shape[1], rgb.shape[0])
        inputs = []
        for h in hands:
            if self.hand_mode != "Both" and h.label != self.hand_mode:
                continue
            inputs.append((h.label, self.renderer.to_screen(h.lm), h.lm[:, 2]))
        self.states, self.two = self.gestures.update(inputs, now, self.screen.get_height())

    def frame(self, dt, now):
        events = []
        if self.camera.running and now - self.last_data_t < 0.4:
            for st in self.states.values():
                events.extend(st.events)
            if self.manual_blast and self.states:
                next(iter(self.states.values())).events.append("thrust")
                events.append("thrust")
            states, two = self.states, self.two
        else:
            states, two = {}, self.gestures.update([], now, 1)[1]
        self.manual_blast = False
        self.engine.update(dt, now, states, two, self.fps)
        for st in self.states.values():
            st.events = []                      # events fire exactly once
        for ev in events:
            if ev in EVENT_NAMES:
                self.flash, self.flash_t = EVENT_NAMES[ev], now

    def overlay_info(self, now):
        lines = []
        live = self.camera.running and now - self.last_data_t < 0.4
        if live and self.states:
            for label in ("Left", "Right"):
                st = self.states.get(label)
                if st is not None:
                    lines.append((f"{label}: {POSE_NAMES[st.pose]}  {int(st.confidence * 100)}%  "
                                  f"hold {st.hold:.1f}s", (255, 225, 250)))
                    if st.speed > config.TRAIL_SPEED:
                        lines.append((f"   fast movement - Energy trail ({st.speed:.1f} H/s)", (255, 150, 230)))
            if self.two is not None and self.two.pose != NONE:
                lines.append((TWO_NAMES[self.two.pose], (255, 120, 220)))
        else:
            lines.append(("No hands detected" if self.camera.running else "Camera off", (170, 140, 185)))
        return lines

    def run(self):
        last = time.monotonic()
        try:
            while self.running:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        self.running = False
                    elif e.type == pygame.VIDEORESIZE:
                        self.screen = pygame.display.get_surface()
                        self._resized()
                    elif e.type == pygame.KEYDOWN:
                        self.on_key(e)
                    else:
                        for a in self.ui.handle(e, self.camera.running):
                            self.do(a)

                now = time.monotonic()
                dt = now - last
                last = now
                self.fps = self.fps * 0.92 + (1.0 / max(dt, 1e-3)) * 0.08

                self.poll_camera(now)
                if self.camera.running:
                    self.frame(dt, now)
                self.screen = pygame.display.get_surface()
                self.renderer.draw_background(self.screen, self.rgb if self.camera.running else None)
                if self.camera.running:
                    self.renderer.draw_effects(self.screen, self.engine)

                info = {
                    "running": self.camera.running, "camera_ok": self.camera.camera_ok,
                    "tracking_ok": self.camera.tracking_ok, "fps": self.fps,
                    "particles": self.engine.particle_count, "adapt": self.engine.adapt,
                    "thrust": self.gestures.thrust_scale, "overlay": self.overlay_info(now),
                    "flash": self.flash if now - self.flash_t < 0.9 else "",
                    "error": self.error if not self.camera.running else (
                        "Starting camera..." if not self.camera.camera_ok and self.rgb is None else None),
                }
                self.ui.draw(self.screen, info)
                pygame.display.flip()
                self.clock.tick(config.TARGET_FPS)
        finally:
            self.camera.stop()          # releases the webcam and MediaPipe
            pygame.quit()


def main():
    App().run()


if __name__ == "__main__":
    main()
