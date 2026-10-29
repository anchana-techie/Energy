"""Compositing: webcam background + additive energy layer + bloom."""
import cv2
import numpy as np
import pygame


class Renderer:
    def __init__(self, size):
        self.W, self.H = size
        self.fx = pygame.Surface(size)
        self._buf = None                  # keeps the frame buffer alive for frombuffer
        self.frame_size = (16, 9)         # (w, h) of last camera frame, for coordinate mapping
        self._small_key = None

    def resize(self, size):
        self.W, self.H = size
        self.fx = pygame.Surface(size)

    # -- coordinate mapping (cover-fit of camera frame into the window) ----------
    def _fit(self):
        fw, fh = self.frame_size
        s = max(self.W / fw, self.H / fh)
        return s, (self.W - fw * s) / 2.0, (self.H - fh * s) / 2.0

    def to_screen(self, lm):
        """Map normalised (n, >=2) landmark coordinates to window pixels."""
        fw, fh = self.frame_size
        s, ox, oy = self._fit()
        out = np.empty((len(lm), 2), np.float32)
        out[:, 0] = lm[:, 0] * fw * s + ox
        out[:, 1] = lm[:, 1] * fh * s + oy
        return out

    # -- drawing -------------------------------------------------------------------
    def draw_background(self, screen, rgb):
        if rgb is None:
            screen.fill((8, 4, 16))
            return
        fh, fw = rgb.shape[:2]
        self.frame_size = (fw, fh)
        s, ox, oy = self._fit()
        sw, sh = max(self.W, int(round(fw * s))), max(self.H, int(round(fh * s)))
        img = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
        x0, y0 = (sw - self.W) // 2, (sh - self.H) // 2
        img = img[y0:y0 + self.H, x0:x0 + self.W]
        img = cv2.convertScaleAbs(img, alpha=0.78, beta=-6)     # slight darkening so glow pops
        self._buf = np.ascontiguousarray(img)
        surf = pygame.image.frombuffer(self._buf, (self.W, self.H), "RGB")
        screen.blit(surf, (0, 0))

    def draw_effects(self, screen, engine, gain=1.0):
        W, H = self.W, self.H
        fx = self.fx
        fx.fill((0, 0, 0))
        engine.draw(fx)

        ds = engine.ds
        lw, lh = max(2, W // ds), max(2, H // ds)
        small = pygame.transform.smoothscale(fx, (lw, lh))
        parts = engine.particle_buffer(lw, lh)
        arr = pygame.surfarray.pixels3d(small)
        arr[:] = np.clip(arr.astype(np.float32) * gain + parts.swapaxes(0, 1), 0, 255).astype(np.uint8)
        del arr
        wide_small = pygame.transform.smoothscale(small, (max(2, lw // 3), max(2, lh // 3)))
        wide_small.fill((150, 150, 150), special_flags=pygame.BLEND_RGB_MULT)
        small.fill((200, 200, 200), special_flags=pygame.BLEND_RGB_MULT)
        bloom = pygame.transform.smoothscale(small, (W, H))
        wide = pygame.transform.smoothscale(wide_small, (W, H))

        screen.blit(fx, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        screen.blit(bloom, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        screen.blit(wide, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
