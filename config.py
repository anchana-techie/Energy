"""Central configuration: colors, performance presets and gesture thresholds."""

APP_TITLE = "Gwen Tennyson - Anodite Energy Simulator"

# --- Camera -----------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
TRACK_WIDTH = 480            # frames are downscaled to this width for MediaPipe
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_FILE = "hand_landmarker.task"   # only used if mp.solutions is unavailable

# --- Window -----------------------------------------------------------------
WINDOW_SIZE = (1280, 720)
TARGET_FPS = 60

# --- Colors (RGB) -----------------------------------------------------------
WHITE_HOT = (255, 240, 255)
PINK_LIGHT = (255, 150, 235)
PINK = (255, 70, 200)
MAGENTA = (220, 30, 190)
VIOLET = (130, 40, 220)
DEEP_VIOLET = (60, 15, 140)
# Life gradient for particles: young -> old
PARTICLE_GRADIENT = [WHITE_HOT, PINK_LIGHT, PINK, VIOLET, DEEP_VIOLET]

# --- Particle quality presets: (max particles, bloom downscale) -------------
QUALITY_PRESETS = {
    "Low": (1500, 8),
    "Medium": (3500, 6),
    "High": (7000, 5),
    "Ultra": (12000, 4),
}
QUALITY_ORDER = ["Low", "Medium", "High", "Ultra"]
DEFAULT_QUALITY = "Medium"
ADAPTIVE_MIN_SCALE = 0.25     # lowest particle-emission multiplier when FPS drops
ADAPTIVE_LOW_FPS = 24
ADAPTIVE_HIGH_FPS = 48

# --- Tracking ---------------------------------------------------------------
MAX_HANDS = 2
DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.5
LANDMARK_SMOOTHING = 0.55     # EMA weight of the *new* sample (higher = snappier)
HAND_LOST_GRACE = 0.25        # seconds a hand is kept alive after tracking drops

# --- Gesture thresholds (distances are in units of window height) -----------
GESTURE_SCORE_EMA = 0.45      # temporal smoothing of per-gesture scores
GESTURE_ENTER = 0.62          # score needed to enter a static gesture
GESTURE_KEEP = 0.45           # score needed to *stay* in it (hysteresis)
GESTURE_MIN_FRAMES = 3        # consecutive frames a challenger must win

VELOCITY_WINDOW = 0.12        # seconds used for velocity estimates

# Forward thrust: growth rate of apparent hand size, d(ln size)/dt (1/s)
THRUST_GROWTH_RATE = 1.15
THRUST_MAX_LATERAL = 1.6      # reject if lateral speed (H/s) exceeds this
THRUST_COOLDOWN = 1.0         # seconds between blasts per hand
THRUST_SENSITIVITY_STEP = 0.15

SWEEP_SPEED = 1.7             # horizontal speed (H/s) to trigger the whip
SWEEP_AXIS_RATIO = 1.8        # |vx| must exceed |vy| * ratio
SWEEP_COOLDOWN = 0.45
TRAIL_SPEED = 0.9             # speed (H/s) where trails begin
TRAIL_SPEED_MAX = 3.5

RAISED_Y = 0.48               # wrists above this fraction of height = "raised"
APART_DIST = 0.75             # hand distance (in H) above which "apart" holds
TOGETHER_DIST = 0.55          # hand distance below which "together" holds
TWO_HAND_HOLD = 0.12          # seconds a two-hand pose must persist

CHARGE_FULL_TIME = 4.0        # seconds of holding open palm for max aura

# Palm orientation: score above ENTER shows the palm to the camera; below LEAVE hides energy
FACING_ENTER = 0.18
FACING_LEAVE = 0.06
ANCHOR_DISTANCE = 0.75        # energy hovers this many palm-sizes beyond the palm centre (0 = centred on palm)
