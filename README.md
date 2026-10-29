# Anodite Energy Simulator (Gwen Tennyson powers)

Real-time webcam app: your hand movements control pink Anodite energy effects drawn over your live video.
Everything runs locally; webcam frames are never saved or uploaded.

## Project structure

```
Energy/
├── main.py                 # entry point, app loop, keyboard shortcuts
├── camera.py               # webcam capture + tracking thread
├── hand_tracking.py        # MediaPipe hands, landmark smoothing, lost-hand recovery
├── gesture_recognition.py  # poses, velocity/acceleration, thrust/sweep, two-hand poses
├── energy_effects.py       # particles, orbs, beams, whips, trails, shockwaves, projectiles
├── renderer.py             # webcam background + additive effect layer + bloom
├── ui.py                   # buttons, slider, status, guide, overlays
├── config.py               # colors, quality presets, gesture thresholds
├── requirements.txt
└── README.md
```

## Install (Windows PowerShell)

MediaPipe does not support Python 3.14, so use Python 3.10 or 3.11:

```powershell
cd D:\projects\Energy
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked: `Set-ExecutionPolicy -Scope Process Bypass`.

## Run

```powershell
python main.py
```

Click **START MAGIC** (or press Enter). Windows asks for camera access the first time.

## Controls

| Key | Action |
|---|---|
| Enter | Start camera |
| F / F11 | Fullscreen cinematic mode (UI panel hidden) |
| ESC | Exit fullscreen; press again to quit safely |
| Q | Quit |
| H | Show/hide panel |
| G | Gesture guide |
| R | Emergency reset (clear all effects) |
| Space | Manual energy blast (thrust fallback) |
| `[` / `]` | Lower / raise forward-thrust sensitivity |

The panel has STOP CAMERA, RESET EFFECTS, power intensity slider, particle quality (Low–Ultra),
Left/Right/Both hand selector, camera/tracking status and FPS.

## Gestures

| Gesture | Power |
|---|---|
| Open palm | Charge: palm orb, aura grows the longer you hold (about 4 s to full) |
| Close fist | Compress: smaller, brighter orb, sparks, faster pulse. Opening after a long fist releases a burst |
| Push hand toward the camera | Energy blast (projectile, trail, shockwave, impact) |
| Fast sideways sweep | Energy whip ribbon |
| Both hands raised, palms open | Maximum power aura, body glow, energy waves |
| Hands far apart | Expansion field with strands between palms |
| Hands close together | Energy orb; bigger and brighter the closer and longer |
| Index finger out, others curled | Precision beam from the fingertip |
| Any fast movement | Flowing trails; speed sets length and brightness |

Pose recognition uses finger-extension ratios from the 21 MediaPipe landmarks, smoothed over time with
hysteresis. Blasts have a 1 s cooldown per hand and need several frames of consistent motion.

## Calibration guide

A webcam has no true depth sensor. The **forward thrust** is estimated from how fast the apparent
hand size grows (`d ln(size)/dt`) while the hand is not sliding sideways.

1. Stand about 0.5–1 m from the camera in good, even light with your hand fully visible.
2. Push your open hand quickly toward the screen. "ENERGY BLAST!" should appear.
3. Not firing? Press `]` to make it easier. Firing by accident? Press `[`.
4. If it is still unreliable, use **Space** as the documented alternative, or tune
   `THRUST_GROWTH_RATE`, `THRUST_MAX_LATERAL`, `THRUST_COOLDOWN` in `config.py`.
5. Whip too sensitive or too hard: change `SWEEP_SPEED`. Two-hand poses: `RAISED_Y`, `APART_DIST`,
   `TOGETHER_DIST`. Distances are in units of window height.

## Troubleshooting

| Problem | Fix |
|---|---|
| "No webcam found, or access was denied" | Close Teams/Zoom/browser tabs using the camera. Windows Settings > Privacy & security > Camera: allow desktop apps. Try `CAMERA_INDEX = 1` in `config.py` |
| `Missing Python packages` | Activate the venv and run `pip install -r requirements.txt` |
| `pip` cannot find mediapipe | You are on an unsupported Python (3.14). Use 3.10/3.11 as above |
| Hands not detected | Better lighting, plain background, keep the whole hand in frame |
| Wrong hand labels / effects on the wrong hand | Use the Both/Left/Right selector; the mirrored view is intentional |
| Hand tracking error about `hand_landmarker.task` | Newer MediaPipe builds need that model; the app downloads it once, or download it manually from the URL in `config.py` and place it next to `main.py` |
| Black window / crash on start | Update graphics drivers; try `pip install --upgrade pygame` |

## Performance tips for ordinary laptops

- Set particle quality to **Low** or **Medium**. The app also adapts emission automatically when FPS drops below 24.
- Close other apps using the camera or GPU; plug in the laptop and use the high-performance power plan.
- Use a smaller window instead of fullscreen. Bloom cost scales with resolution.
- Lower `CAMERA_WIDTH/HEIGHT` (e.g. 640x480) and `TRACK_WIDTH` in `config.py`.
- Lower the power intensity slider to reduce particle emission.

## Privacy

Frames are held in memory only for the latest frame, processed locally, and discarded. Nothing is written to disk or sent anywhere
(the optional one-time model download in the fallback path is the only network use).
