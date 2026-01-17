from __future__ import annotations

import cv2
from pathlib import Path
from typing import Dict, List

from src.config.settings import FRAME_SAMPLE_RATE
from src.config.paths import FRAMES_DIR
from src.utils.logger import get_logger, ProgressTracker, log_section

log = get_logger("frame-extractor")


# --------------------------------------------------
# QUALITY CHECKS
# --------------------------------------------------

def _is_blurry(image, threshold: float = 100.0) -> bool:
    """
    Simple blur detection using variance of Laplacian.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def extract_frames(
    video_path: Path,
    video_id: str,
) -> Dict:
    """
    Extract sampled frames from a video.

    Returns:
        {
            "video_id": str,
            "total_frames": int,
            "extracted_frames": int,
            "frames": List[Path]
        }
    """

    log_section("Frame Extraction")
    log.info(f"Video: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir = FRAMES_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_frames: List[Path] = []
    frame_idx = 0
    saved_count = 0

    with ProgressTracker(
        title="Extracting frames",
        total=total_frames,
    ) as progress:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_SAMPLE_RATE == 0:
                if not _is_blurry(frame):
                    frame_name = f"frame_{frame_idx:06d}.jpg"
                    frame_path = output_dir / frame_name

                    cv2.imwrite(str(frame_path), frame)
                    saved_frames.append(frame_path)
                    saved_count += 1

            frame_idx += 1
            progress.advance()

    cap.release()

    log.info(
        f"Extracted {saved_count} frames "
        f"from {total_frames} total frames"
    )

    return {
        "video_id": video_id,
        "total_frames": total_frames,
        "extracted_frames": saved_count,
        "frames": saved_frames,
    }
