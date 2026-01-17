from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2

from src.config.paths import CROPS_DIR
from src.utils.logger import get_logger, log_section

log = get_logger("cropper")


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def _expand_bbox(
    bbox: List[int],
    img_width: int,
    img_height: int,
    padding_ratio: float = 0.15,
) -> List[int]:
    """
    Expand bounding box by padding_ratio on all sides.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)

    nx1 = max(0, x1 - pad_w)
    ny1 = max(0, y1 - pad_h)
    nx2 = min(img_width, x2 + pad_w)
    ny2 = min(img_height, y2 + pad_h)

    return [nx1, ny1, nx2, ny2]


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def crop_items(
    tracked_items: List[Dict],
    video_id: str,
) -> List[Dict]:
    """
    Crop detected items from their best frames.

    Returns:
        [
          {
            "item": str,
            "confidence": float,
            "crop_path": Path,
            "bbox": [x1, y1, x2, y2]
          }
        ]
    """

    log_section("Cropping Detected Items")

    output_dir = CROPS_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    cropped_items: List[Dict] = []

    for idx, item in enumerate(tracked_items):
        frame_path: Path = item["best_frame"]
        bbox = item["bbox"]

        image = cv2.imread(str(frame_path))
        if image is None:
            log.warning(f"Could not read frame: {frame_path}")
            continue

        h, w = image.shape[:2]
        x1, y1, x2, y2 = _expand_bbox(bbox, w, h)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            log.warning("Empty crop skipped")
            continue

        crop_name = (
            f"{item['item']}_{idx:02d}_"
            f"{frame_path.stem}.jpg"
        )
        crop_path = output_dir / crop_name

        cv2.imwrite(str(crop_path), crop)

        cropped_items.append(
            {
                "item": item["item"],
                "confidence": item["confidence"],
                "crop_path": crop_path,
                "bbox": [x1, y1, x2, y2],
            }
        )

    log.info(f"Saved {len(cropped_items)} cropped items")

    return cropped_items
