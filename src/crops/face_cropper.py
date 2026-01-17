from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2

from src.utils.logger import get_logger

log = get_logger("face-cropper")


def crop_face_from_person(
    frame_path: Path,
    person_bbox: list[int],
    output_path: Path,
) -> Optional[Path]:
    """
    Geometry-based face crop from person bbox.
    """

    img = cv2.imread(str(frame_path))
    if img is None:
        return None

    h, w, _ = img.shape
    x1, y1, x2, y2 = person_bbox

    pw = x2 - x1
    ph = y2 - y1

    # ---- face heuristic ----
    face_h = int(ph * 0.28)
    face_y1 = y1 + int(ph * 0.05)
    face_y2 = face_y1 + face_h

    face_x1 = x1 + int(pw * 0.15)
    face_x2 = x2 - int(pw * 0.15)

    # clamp
    face_x1 = max(0, face_x1)
    face_y1 = max(0, face_y1)
    face_x2 = min(w, face_x2)
    face_y2 = min(h, face_y2)

    if face_x2 <= face_x1 or face_y2 <= face_y1:
        return None

    face = img[face_y1:face_y2, face_x1:face_x2]

    if face.size == 0:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), face)

    return output_path
