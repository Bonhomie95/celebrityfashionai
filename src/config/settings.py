from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# --------------------------------------------------
# APP
# --------------------------------------------------

APP_NAME = os.getenv("APP_NAME", "celebrity-fashion-ai")
ENV = os.getenv("ENV", "development")
DEBUG = ENV != "production"

# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATA_DIR = BASE_DIR / "data"
RAW_VIDEO_DIR = DATA_DIR / "raw_videos"
FRAMES_DIR = DATA_DIR / "frames"
CROPS_DIR = DATA_DIR / "crops"
METADATA_DIR = DATA_DIR / "metadata"
CACHE_DIR = DATA_DIR / "cache"

OUTPUT_DIR = BASE_DIR / "outputs"
TAGGED_VIDEO_DIR = OUTPUT_DIR / "tagged_videos"

MODEL_DIR = BASE_DIR / "models"

# Ensure directories exist
for path in [
    RAW_VIDEO_DIR,
    FRAMES_DIR,
    CROPS_DIR,
    METADATA_DIR,
    CACHE_DIR,
    TAGGED_VIDEO_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# VIDEO INGESTION
# --------------------------------------------------

MAX_VIDEO_DURATION_SEC = int(os.getenv("MAX_VIDEO_DURATION_SEC", "90"))
FRAME_SAMPLE_RATE = int(os.getenv("FRAME_SAMPLE_RATE", "5"))  # every N frames
MIN_VIDEO_RESOLUTION = int(os.getenv("MIN_VIDEO_RESOLUTION", "720"))

# --------------------------------------------------
# DETECTION (YOLO)
# --------------------------------------------------

YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    str(MODEL_DIR / "yolo" / "weights" / "yolov8-fashion.pt"),
)

YOLO_CONFIDENCE_THRESHOLD = float(
    os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.5")
)

YOLO_IOU_THRESHOLD = float(
    os.getenv("YOLO_IOU_THRESHOLD", "0.45")
)

DETECTION_CLASSES = [
    "shoe",
    "watch",
    "necklace",
    "ring",
    "bracelet",
]

# --------------------------------------------------
# BRAND & PRICE ESTIMATION
# --------------------------------------------------

PRICE_CONFIDENCE_MIN = float(
    os.getenv("PRICE_CONFIDENCE_MIN", "0.6")
)

DEFAULT_PRICE_RANGE = "$500 – $5,000"

ENABLE_WEB_LOOKUP = os.getenv("ENABLE_WEB_LOOKUP", "false").lower() == "true"

# --------------------------------------------------
# SEARCH / APIs (OPTIONAL)
# --------------------------------------------------

BING_SEARCH_API_KEY = os.getenv("BING_SEARCH_API_KEY")
BING_SEARCH_ENDPOINT = os.getenv(
    "BING_SEARCH_ENDPOINT",
    "https://api.bing.microsoft.com/v7.0/images/search",
)

# --------------------------------------------------
# RENDERING
# --------------------------------------------------

VIDEO_CODEC = os.getenv("VIDEO_CODEC", "libx264")
VIDEO_BITRATE = os.getenv("VIDEO_BITRATE", "6M")
TEXT_FONT = os.getenv("TEXT_FONT", "assets/fonts/Inter-Bold.ttf")

# --------------------------------------------------
# LOGGING
# --------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

def validate_settings() -> None:
    if not YOLO_MODEL_PATH:
        raise RuntimeError("YOLO_MODEL_PATH is not set")

    if ENABLE_WEB_LOOKUP and not BING_SEARCH_API_KEY:
        raise RuntimeError(
            "ENABLE_WEB_LOOKUP=true but BING_SEARCH_API_KEY is missing"
        )


validate_settings()
