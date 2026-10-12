from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best.pt"
INPUT_VIDEO = BASE_DIR / "input" / "traffic.mp4"
OUTPUT_DIR = BASE_DIR / "output"


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = str(MODEL_PATH)

# YOLO inference image size
IMG_SIZE = 640

# Minimum confidence
CONF_THRESHOLD = 0.35

# NMS IoU threshold
IOU_THRESHOLD = 0.45


# ============================================================
# EDGE PERFORMANCE
# ============================================================

# Process only every Nth frame.
# 1 = every frame
# 2 = every second frame
FRAME_SKIP = 1

# Maximum inference FPS.
# Set None for maximum possible speed.
MAX_INFERENCE_FPS = None


# ============================================================
# VEHICLE CLASSES
# ============================================================

VEHICLE_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# If your trained model uses different class IDs,
# change the dictionary above.


# ============================================================
# TRACKING
# ============================================================

TRACKER = "bytetrack.yaml"

TRACK_CONF = 0.35

TRACK_IOU = 0.5

TRACK_BUFFER = 30


# ============================================================
# DISPLAY
# ============================================================

SHOW_WINDOW = True

SAVE_OUTPUT = True