from __future__ import annotations
from pathlib import Path


# --------------------------------------------------
# BASE
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

# --------------------------------------------------
# DATA
# --------------------------------------------------

DATA_DIR = BASE_DIR / "data"

RAW_VIDEO_DIR = DATA_DIR / "raw_videos"
FRAMES_DIR = DATA_DIR / "frames"
CROPS_DIR = DATA_DIR / "crops"
METADATA_DIR = DATA_DIR / "metadata"
CACHE_DIR = DATA_DIR / "cache"

# --------------------------------------------------
# MODELS
# --------------------------------------------------

MODEL_DIR = BASE_DIR / "models"
YOLO_DIR = MODEL_DIR / "yolo"
YOLO_WEIGHTS_DIR = YOLO_DIR / "weights"
YOLO_CONFIG_DIR = YOLO_DIR / "configs"

# --------------------------------------------------
# OUTPUTS
# --------------------------------------------------

OUTPUT_DIR = BASE_DIR / "outputs"
TAGGED_VIDEO_DIR = OUTPUT_DIR / "tagged_videos"
REPORTS_DIR = OUTPUT_DIR / "reports"

# --------------------------------------------------
# ASSETS
# --------------------------------------------------

ASSETS_DIR = BASE_DIR / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
BACKGROUNDS_DIR = ASSETS_DIR / "backgrounds"

# --------------------------------------------------
# TEMP
# --------------------------------------------------

TMP_DIR = BASE_DIR / "tmp"

# --------------------------------------------------
# INIT
# --------------------------------------------------

ALL_DIRS = [
    RAW_VIDEO_DIR,
    FRAMES_DIR,
    CROPS_DIR,
    METADATA_DIR,
    CACHE_DIR,
    YOLO_WEIGHTS_DIR,
    YOLO_CONFIG_DIR,
    TAGGED_VIDEO_DIR,
    REPORTS_DIR,
    FONTS_DIR,
    BACKGROUNDS_DIR,
    TMP_DIR,
]


def ensure_directories() -> None:
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


# Run once on import
ensure_directories()
