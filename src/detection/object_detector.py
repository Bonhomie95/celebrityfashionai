from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set, Union, Optional

import cv2
from ultralytics.models import YOLO 

from src.config.settings import (
    YOLO_MODEL_PATH,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    DETECTION_CLASSES,
)
from src.utils.logger import get_logger, log_section, ProgressTracker

log = get_logger("object-detector")


NamesType = Union[Dict[int, str], List[str]]


def _normalize_names(names: NamesType) -> Dict[int, str]:
    """
    Ultralytics model.names can be a dict {id: name} or a list [name,...].
    Normalize to dict[int, str].
    """
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


class FashionObjectDetector:
    """
    YOLO-based fashion item detector.
    Instantiate ONCE per pipeline run.
    """

    def __init__(self) -> None:
        log.info("Loading YOLO model...")
        self.model = YOLO(YOLO_MODEL_PATH)

        self.class_names: Dict[int, str] = _normalize_names(self.model.names)  # type: ignore[arg-type]
        allowed = {c.lower() for c in DETECTION_CLASSES}

        self.allowed_class_ids: Set[int] = {
            idx for idx, name in self.class_names.items() if name.lower() in allowed
        }

        enabled = [self.class_names[i] for i in sorted(self.allowed_class_ids)]
        log.info(f"Enabled classes: {enabled}")

    def detect(self, frames: List[Path]) -> List[Dict[str, Any]]:
        """
        Run detection on extracted frames.

        Returns list of:
          {
            "frame": Path,
            "item": str,
            "confidence": float,
            "bbox": [x1, y1, x2, y2]
          }
        """
        log_section("Object Detection")
        detections: List[Dict[str, Any]] = []

        if not frames:
            log.warning("No frames provided to detector.")
            return detections

        with ProgressTracker(
            title="Running YOLO inference", total=len(frames)
        ) as progress:
            for frame_path in frames:
                image = cv2.imread(str(frame_path))
                if image is None:
                    progress.advance()
                    continue

                results = self.model.predict(
                    source=image,
                    conf=YOLO_CONFIDENCE_THRESHOLD,
                    iou=YOLO_IOU_THRESHOLD,
                    verbose=False,
                )

                # ✅ Fix: results can be None or empty
                if not results:
                    progress.advance()
                    continue

                for result in results:
                    boxes = getattr(result, "boxes", None)

                    # ✅ Fix: boxes can be None (then it's not iterable)
                    if boxes is None or len(boxes) == 0:
                        continue

                    for box in boxes:
                        cls_tensor = getattr(box, "cls", None)
                        conf_tensor = getattr(box, "conf", None)
                        xyxy_tensor = getattr(box, "xyxy", None)

                        if (
                            cls_tensor is None
                            or conf_tensor is None
                            or xyxy_tensor is None
                        ):
                            continue

                        cls_id = int(cls_tensor[0])
                        if cls_id not in self.allowed_class_ids:
                            continue

                        conf = float(conf_tensor[0])

                        x1, y1, x2, y2 = map(int, xyxy_tensor[0].tolist())

                        detections.append(
                            {
                                "frame": frame_path,
                                "item": self.class_names.get(cls_id, str(cls_id)),
                                "confidence": conf,
                                "bbox": [x1, y1, x2, y2],
                            }
                        )

                progress.advance()

        log.info(f"Detected {len(detections)} items total")
        return detections
