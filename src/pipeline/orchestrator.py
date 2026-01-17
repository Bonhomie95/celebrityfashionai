from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict

from src.utils.logger import get_logger, log_section
from src.ingestion.video_downloader import download_video
from src.processing.frame_extractor import extract_frames
from src.detection.object_detector import FashionObjectDetector
from src.detection.item_tracker import track_items

from src.crops.cropper import crop_items
from src.crops.quality_check import filter_crops

from src.crops.face_cropper import crop_face_from_person
from src.classification.glasses_classifier import classify_glasses

from src.enrichment.price_estimator import estimate_prices
from src.video.overlay import render_overlay

from src.config.paths import FACE_DIR

log = get_logger("orchestrator")


# --------------------------------------------------
# ORCHESTRATOR
# --------------------------------------------------


def run(
    url: str,
    *,
    skip_overlay: bool = False,
) -> Optional[Path]:
    """
    Run the full celebrity fashion pipeline on a single video.
    """

    log_section("PIPELINE START")

    # --------------------------------------------------
    # 1. DOWNLOAD
    # --------------------------------------------------

    video_meta: Dict = download_video(url)
    video_id = video_meta["video_id"]
    video_path: Path = video_meta["path"]

    # --------------------------------------------------
    # 2. FRAME EXTRACTION
    # --------------------------------------------------

    frame_data = extract_frames(
        video_path=video_path,
        video_id=video_id,
    )

    if not frame_data["frames"]:
        log.warning("No frames extracted — stopping pipeline")
        return None

    # --------------------------------------------------
    # 3. OBJECT DETECTION
    # --------------------------------------------------

    detector = FashionObjectDetector()
    detections = detector.detect(frame_data["frames"])

    if not detections:
        log.warning("No fashion items detected — stopping pipeline")
        return None

    # --------------------------------------------------
    # 4. ITEM TRACKING (DEDUP)
    # --------------------------------------------------

    unique_items = track_items(detections)

    if not unique_items:
        log.warning("All detections deduplicated away — stopping")
        return None

    # --------------------------------------------------
    # 4.5 FACE CROPPING (PERSON ONLY)
    # --------------------------------------------------

    for item in unique_items:
        if item.get("item") != "person":
            continue

        face_path = crop_face_from_person(
            frame_path=item["frame"],
            person_bbox=item["bbox"],
            output_path=FACE_DIR / f"{item['id']}_face.jpg",
        )

        if face_path:
            item["face_crop"] = face_path

    # --------------------------------------------------
    # 5. CROPPING (FASHION ITEMS)
    # --------------------------------------------------

    crops = crop_items(
        tracked_items=unique_items,
        video_id=video_id,
    )

    if not crops:
        log.warning("No crops created — stopping")
        return None

    # --------------------------------------------------
    # 6. QUALITY FILTER
    # --------------------------------------------------

    good_crops, rejected = filter_crops(crops)

    if not good_crops:
        log.warning("All crops failed quality checks — stopping")
        return None

    # --------------------------------------------------
    # 6.5 GLASSES CLASSIFICATION (FACE ONLY)
    # --------------------------------------------------

    for item in good_crops:
        face = item.get("face_crop")
        if not face:
            continue

        item["glasses"] = classify_glasses(face)

    # --------------------------------------------------
    # 7. PRICE ESTIMATION
    # --------------------------------------------------

    priced_items = estimate_prices(good_crops)

    if skip_overlay:
        log.info("skip_overlay=True → pipeline ends after price estimation")
        return None

    # --------------------------------------------------
    # 8. VIDEO OVERLAY
    # --------------------------------------------------

    final_video = render_overlay(
        video_path=video_path,
        video_id=video_id,
        priced_items=priced_items,
    )

    log_section("PIPELINE COMPLETE")
    log.info(f"Final video → {final_video}")

    return final_video
