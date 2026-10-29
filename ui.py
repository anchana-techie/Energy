"""Interface: buttons, slider, selectors, status indicators, gesture guide and overlays."""
import pygame

import config

ACCENT = (255, 80, 205)
ACCENT_DIM = (150, 40, 130)
PANEL = (20, 8, 34, 175)
TEXT = (250, 225, 250)
MUTED = (170, 140, 185)
GOOD = (120, 255, 170)
BAD = (255, 90, 110)

GUIDE = [
    ("Open palm", "Charge energy orb (hold = stronger)"),
    ("Close fist", "Compress energy, sparks + pulse"),
    ("Push hand forward", "Energy blast (SPACE = manual)"),
    ("Fast sideways sweep", "Energy whip"),
    ("Both hands raised", "Maximum Anodite power"),
    ("Hands far apart", "Energy expansion field"),
    ("Hands together", "Energy orb between palms"),
    ("Index finger point", "Precision beam"),
    ("Fast movement", "Energy trails"),
]


class Button:
    def __init__(self, label, action, primary=False):
        self.label, self.action, self.primary = label, action, primary
        self.rect = pygame.Rect(0, 0, 10, 10)
        self.active = False
        self.enabled = True

    def draw(self, surf, font, mouse):
        hover = self.rect.collidepoint(mouse) and self.enabled
        base = ACCENT if (self.primary or self.active) else ACCENT_DIM
        fill = (base[0] // (2 if not hover else 1), base[1] // (2 if not hover else 1), base[2] // (2 if not hover else 1))
        if not self.enabled:
            fill = (50, 40, 60)
        pygame.draw.rect(surf, fill, self.rect, border_radius=8)
        pygame.draw.rect(surf, base if self.enabled else (90, 80, 100), self.rect, 2, border_radius=8)
        img = font.render(self.label, True, TEXT if self.enabled else MUTED)
        surf.blit(img, img.get_rect(center=self.rect.center))


class Slider:
    def __init__(self, label, lo, hi, value):
        self.label, self.lo, self.hi, self.value = label, lo, hi, value
        self.rect = pygame.Rect(0, 0, 10, 10)
        self.drag = False

    def _set_from(self, x):
        u = (x - self.rect.x) / max(1, self.rect.w)
        self.value = self.lo + (self.hi - self.lo) * min(1.0, max(0.0, u))

    def event(self, e):
        hit = self.rect.inflate(0, 18)
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and hit.collidepoint(e.pos):
            self.drag = True
            self._set_from(e.pos[0])
            return True
        if e.type == pygame.MOUSEMOTION and self.drag:
            self._set_from(e.pos[0])
            return True
        if e.type == pygame.MOUSEBUTTONUP and e.button == 1 and self.drag:
            self.drag = False
        return False

    def draw(self, surf, font):
        surf.blit(font.render(f"{self.label}: {int(self.value * 100)}%", True, TEXT), (self.rect.x, self.rect.y - 22))
        pygame.draw.rect(surf, (60, 30, 80), self.rect, border_radius=4)
        u = (self.value - self.lo) / (self.hi - self.lo)
        fill = self.rect.copy()
        fill.w = int(self.rect.w * u)
        pygame.draw.rect(surf, ACCENT, fill, border_radius=4)
        pygame.draw.circle(surf, (255, 240, 255), (self.rect.x + fill.w, self.rect.centery), 9)


class UI:
    PANEL_W = 290

    def __init__(self, size):
        pygame.font.init()
        name = pygame.font.match_font("segoeui,arial,helvetica") or None
        self.f_big = pygame.font.Font(name, 54)
        self.f_title = pygame.font.Font(name, 28)
        self.f = pygame.font.Font(name, 18)
        self.f_s = pygame.font.Font(name, 15)
        self.start = Button("START MAGIC", "start", primary=True)
        self.stop = Button("STOP CAMERA", "stop")
        self.reset = Button("RESET EFFECTS", "reset")
        self.full = Button("FULLSCREEN", "fullscreen")
        self.guide_btn = Button("GUIDE", "guide")
        self.quality_btns = [Button(q, ("quality", q)) for q in config.QUALITY_ORDER]
        self.hand_btns = [Button("Left", ("hands", "Left")), Button("Right", ("hands", "Right")),
                          Button("Both", ("hands", "Both"))]
        self.slider = Slider("Power intensity", 0.2, 2.0, 1.0)
        self.quality, self.hand_mode = config.DEFAULT_QUALITY, "Both"
        self.show_panel = True
        self.show_guide = False
        self.size = size
        self.layout(size)

    def layout(self, size):
        self.size = size
        W, H = size
        x, w = 16, self.PANEL_W - 32
        y = 16
        self.panel_rect = pygame.Rect(0, 0, self.PANEL_W, H)
        for b in (self.start, self.stop):
            b.rect = pygame.Rect(x, y, w, 40)
            y += 48
        self.reset.rect = pygame.Rect(x, y, w, 36)
        y += 44
        self.full.rect = pygame.Rect(x, y, (w - 8) // 2, 34)
        self.guide_btn.rect = pygame.Rect(x + (w - 8) // 2 + 8, y, (w - 8) // 2, 34)
        y += 74
        self.slider.rect = pygame.Rect(x, y, w, 10)
        y += 52
        self._quality_y = y - 26
        bw = (w - 9) // 4
        for i, b in enumerate(self.quality_btns):
            b.rect = pygame.Rect(x + i * (bw + 3), y, bw, 30)
        y += 70
        self._hands_y = y - 26
        bw = (w - 6) // 3
        for i, b in enumerate(self.hand_btns):
            b.rect = pygame.Rect(x + i * (bw + 3), y, bw, 30)
        y += 46
        self._status_y = y
        self.big_start = Button("START MAGIC", "start", primary=True)
        self.big_start.rect = pygame.Rect(0, 0, 340, 64)
        self.big_start.rect.center = (W // 2, H // 2 + 70)

    # -- events -------------------------------------------------------------------
    def handle(self, e, running):
        actions = []
        if e.type == pygame.VIDEORESIZE:
            return actions
        if self.slider.event(e) and self.show_panel and running:
            actions.append(("intensity", self.slider.value))
            return actions
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if not running:
                if self.big_start.rect.collidepoint(e.pos):
                    actions.append("start")
                return actions
            if self.show_panel:
                btns = [self.stop, self.reset, self.full, self.guide_btn] + self.quality_btns + self.hand_btns
                for b in btns:
                    if b.rect.collidepoint(e.pos) and b.enabled:
                        actions.append(b.action)
                        break
        return actions

    # -- drawing ------------------------------------------------------------------
    def _text(self, surf, txt, pos, font=None, color=TEXT):
        surf.blit((font or self.f).render(txt, True, color), pos)

    def _dot(self, surf, x, y, ok, label):
        pygame.draw.circle(surf, GOOD if ok else BAD, (x + 7, y + 10), 7)
        self._text(surf, label, (x + 22, y), self.f_s, TEXT)

    def draw(self, screen, info):
        W, H = screen.get_size()
        if (W, H) != self.size:
            self.layout((W, H))
        mouse = pygame.mouse.get_pos()
        running = info["running"]
        self.stop.enabled = running
        self.start.enabled = not running
        self.quality_btns_update()

        if not running:
            self._draw_title(screen, info, mouse)
            return

        if self.show_panel:
            p = pygame.Surface(self.panel_rect.size, pygame.SRCALPHA)
            p.fill(PANEL)
            screen.blit(p, (0, 0))
            pygame.draw.line(screen, ACCENT_DIM, (self.PANEL_W - 1, 0), (self.PANEL_W - 1, H), 2)
            for b in (self.stop, self.reset, self.full, self.guide_btn):
                b.draw(screen, self.f, mouse)
            self.slider.draw(screen, self.f)
            self._text(screen, "Particle quality", (16, self._quality_y), self.f)
            for b in self.quality_btns:
                b.draw(screen, self.f_s, mouse)
            self._text(screen, "Active hands", (16, self._hands_y), self.f)
            for b in self.hand_btns:
                b.draw(screen, self.f_s, mouse)
            y = self._status_y
            self._dot(screen, 16, y, info["camera_ok"], "Camera connected")
            self._dot(screen, 16, y + 26, info["tracking_ok"], "Hands tracked")
            y += 64
            self._text(screen, f"FPS: {info['fps']:.0f}   Particles: {info['particles']}", (16, y), self.f_s, MUTED)
            self._text(screen, f"Adaptive quality: {int(info['adapt'] * 100)}%", (16, y + 22), self.f_s, MUTED)
            self._text(screen, f"Thrust sensitivity: {info['thrust']:.2f}  ( [ / ] )", (16, y + 44), self.f_s, MUTED)
            self._text(screen, "H: hide panel   G: guide   SPACE: blast", (16, H - 52), self.f_s, MUTED)
            self._text(screen, "F11: fullscreen   ESC: exit", (16, H - 30), self.f_s, MUTED)
        else:
            self._text(screen, f"{info['fps']:.0f} FPS", (W - 90, 12), self.f_s, MUTED)

        self._draw_overlay(screen, info)
        if self.show_guide:
            self._draw_guide(screen)
        if info.get("error"):
            self._banner(screen, info["error"])

    def quality_btns_update(self):
        for b in self.quality_btns:
            b.active = b.action[1] == self.quality
        for b in self.hand_btns:
            b.active = b.action[1] == self.hand_mode

    def _draw_overlay(self, screen, info):
        W, H = screen.get_size()
        lines = info["overlay"]
        x0 = (self.PANEL_W if self.show_panel else 0) + 16
        w = 330
        h = 16 + 22 * len(lines)
        box = pygame.Surface((w, h), pygame.SRCALPHA)
        box.fill((20, 8, 34, 160))
        screen.blit(box, (x0, 14))
        pygame.draw.rect(screen, ACCENT_DIM, (x0, 14, w, h), 1, border_radius=4)
        for i, (txt, col) in enumerate(lines):
            self._text(screen, txt, (x0 + 10, 22 + i * 22), self.f_s, col)
        if info.get("flash"):
            img = self.f_title.render(info["flash"], True, (255, 190, 245))
            screen.blit(img, img.get_rect(center=(W // 2 + (self.PANEL_W // 2 if self.show_panel else 0), H - 60)))

    def _draw_guide(self, screen):
        W, H = screen.get_size()
        w, h = 460, 60 + 30 * len(GUIDE)
        x, y = W - w - 16, 14
        box = pygame.Surface((w, h), pygame.SRCALPHA)
        box.fill((20, 8, 34, 215))
        screen.blit(box, (x, y))
        pygame.draw.rect(screen, ACCENT, (x, y, w, h), 2, border_radius=6)
        self._text(screen, "Gesture guide", (x + 14, y + 10), self.f_title, ACCENT)
        for i, (g, d) in enumerate(GUIDE):
            self._text(screen, g, (x + 14, y + 54 + i * 30), self.f, TEXT)
            self._text(screen, d, (x + 190, y + 56 + i * 30), self.f_s, MUTED)

    def _banner(self, screen, msg, cy=None):
        W, H = screen.get_size()
        lines = msg.split("\n")
        h = 24 + 24 * len(lines)
        box = pygame.Surface((min(W - 40, 820), h), pygame.SRCALPHA)
        box.fill((90, 10, 30, 220))
        r = box.get_rect(center=(W // 2, cy if cy is not None else H // 2))
        screen.blit(box, r)
        for i, ln in enumerate(lines):
            self._text(screen, ln, (r.x + 14, r.y + 12 + i * 24), self.f_s, (255, 230, 235))

    def _draw_title(self, screen, info, mouse):
        W, H = screen.get_size()
        shade = pygame.Surface((W, H), pygame.SRCALPHA)
        shade.fill((12, 4, 24, 215))
        screen.blit(shade, (0, 0))
        t = self.f_big.render("ANODITE ENERGY", True, ACCENT)
        screen.blit(t, t.get_rect(center=(W // 2, H // 2 - 120)))
        s = self.f_title.render("Gwen Tennyson power simulator", True, TEXT)
        screen.blit(s, s.get_rect(center=(W // 2, H // 2 - 65)))
        m = self.f_s.render("Your webcam is processed locally. Nothing is recorded or uploaded.", True, MUTED)
        screen.blit(m, m.get_rect(center=(W // 2, H // 2 - 25)))
        self.big_start.draw(screen, self.f_title, mouse)
        if info.get("error"):
            self._banner(screen, info["error"], H // 2 + 190)
        k = self.f_s.render("Keys: ENTER start   ESC quit", True, MUTED)
        screen.blit(k, k.get_rect(center=(W // 2, H - 40)))
