from __future__ import annotations

from typing import Dict, List
from collections import defaultdict

from src.utils.logger import get_logger, log_section

log = get_logger("item-tracker")


# --------------------------------------------------
# IOU
# --------------------------------------------------


def _iou(box1: List[int], box2: List[int]) -> float:
    """
    Intersection over Union between two boxes.
    Boxes are [x1, y1, x2, y2]
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter_area / float(box1_area + box2_area - inter_area)


# --------------------------------------------------
# TRACKER
# --------------------------------------------------


def track_items(
    detections: List[Dict],
    iou_threshold: float = 0.5,
) -> List[Dict]:
    """
    Merge detections across frames into unique items.

    Input:
        detections: output from object_detector

    Output:
        [
          {
            "id": str,
            "item": str,
            "frame": Path,
            "confidence": float,
            "bbox": [x1, y1, x2, y2],
            "frames_seen": int
          }
        ]
    """

    log_section("Item Tracking & Deduplication")

    grouped: Dict[str, List[Dict]] = defaultdict(list)

    # --------------------------------------------------
    # GROUP BY ITEM TYPE
    # --------------------------------------------------

    for det in detections:
        grouped[det["item"]].append(det)

    final_items: List[Dict] = []

    # --------------------------------------------------
    # PROCESS EACH ITEM TYPE
    # --------------------------------------------------

    for item_name, item_detections in grouped.items():
        clusters: List[List[Dict]] = []

        for det in item_detections:
            matched = False

            for cluster in clusters:
                if _iou(det["bbox"], cluster[0]["bbox"]) >= iou_threshold:
                    cluster.append(det)
                    matched = True
                    break

            if not matched:
                clusters.append([det])

        # --------------------------------------------------
        # PICK BEST FROM EACH CLUSTER
        # --------------------------------------------------

        for idx, cluster in enumerate(clusters):
            best = max(cluster, key=lambda d: d["confidence"])

            final_items.append(
                {
                    "id": f"{item_name}_{idx}",  # ✅ stable ID
                    "item": item_name,
                    "frame": best["frame"],  # ✅ expected downstream
                    "confidence": float(best["confidence"]),
                    "bbox": best["bbox"],
                    "frames_seen": len(cluster),
                }
            )

    log.info(f"Reduced {len(detections)} detections → {len(final_items)} items")

    return final_items
