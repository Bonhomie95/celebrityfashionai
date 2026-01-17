from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger, log_section

log = get_logger("quality-check")


# --------------------------------------------------
# DEFAULT THRESHOLDS (TUNABLE)
# --------------------------------------------------

MIN_WIDTH = 64
MIN_HEIGHT = 64
MIN_AREA = 64 * 64

BLUR_THRESHOLD = 90.0          # Laplacian variance
MIN_CONTRAST_STD = 15.0        # grayscale std-dev
MAX_BLACK_RATIO = 0.6          # too dark / empty


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def _blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _contrast_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.std())


def _black_ratio(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_pixels = np.sum(gray < 15)
    return black_pixels / gray.size


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def filter_crops(
    cropped_items: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter cropped images by quality.

    Returns:
        (accepted, rejected)

    Each rejected item contains a `reason` key.
    """

    log_section("Crop Quality Check")

    accepted: List[Dict] = []
    rejected: List[Dict] = []

    for item in cropped_items:
        crop_path: Path = item["crop_path"]

        image = cv2.imread(str(crop_path))
        if image is None:
            item["reason"] = "unreadable_image"
            rejected.append(item)
            continue

        h, w = image.shape[:2]
        area = h * w

        # ------------------------------------------
        # SIZE CHECK
        # ------------------------------------------

        if w < MIN_WIDTH or h < MIN_HEIGHT or area < MIN_AREA:
            item["reason"] = "too_small"
            rejected.append(item)
            continue

        # ------------------------------------------
        # BLUR CHECK
        # ------------------------------------------

        blur = _blur_score(image)
        if blur < BLUR_THRESHOLD:
            item["reason"] = "too_blurry"
            item["blur_score"] = round(blur, 2)
            rejected.append(item)
            continue

        # ------------------------------------------
        # CONTRAST CHECK
        # ------------------------------------------

        contrast = _contrast_score(image)
        if contrast < MIN_CONTRAST_STD:
            item["reason"] = "low_contrast"
            item["contrast"] = round(contrast, 2)
            rejected.append(item)
            continue

        # ------------------------------------------
        # BLACK / EMPTY CHECK
        # ------------------------------------------

        black_ratio = _black_ratio(image)
        if black_ratio > MAX_BLACK_RATIO:
            item["reason"] = "mostly_dark"
            item["black_ratio"] = round(black_ratio, 2)
            rejected.append(item)
            continue

        # ------------------------------------------
        # ACCEPT
        # ------------------------------------------

        item["quality"] = {
            "blur": round(blur, 2),
            "contrast": round(contrast, 2),
            "black_ratio": round(black_ratio, 2),
        }

        accepted.append(item)

    log.info(
        f"Accepted {len(accepted)} / "
        f"Rejected {len(rejected)} crops"
    )

    return accepted, rejected
