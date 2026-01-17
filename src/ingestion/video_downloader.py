from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Any, Mapping

import yt_dlp

from src.config.settings import (
    MAX_VIDEO_DURATION_SEC,
    MIN_VIDEO_RESOLUTION,
)
from src.config.paths import RAW_VIDEO_DIR
from src.utils.logger import get_logger, log_section, log_kv

log = get_logger("video-downloader")


# --------------------------------------------------
# HELPERS
# --------------------------------------------------


def _video_exists(video_id: str) -> Optional[Path]:
    for file in RAW_VIDEO_DIR.glob(f"{video_id}.*"):
        return file
    return None


def _validate_video_info(info: Mapping[str, Any]) -> None:
    duration = info.get("duration")
    height = info.get("height")

    if isinstance(duration, (int, float)) and duration > MAX_VIDEO_DURATION_SEC:
        raise ValueError(f"Video too long ({duration}s > {MAX_VIDEO_DURATION_SEC}s)")

    if isinstance(height, int) and height < MIN_VIDEO_RESOLUTION:
        raise ValueError(
            f"Video resolution too low ({height}p < {MIN_VIDEO_RESOLUTION}p)"
        )


# --------------------------------------------------
# MAIN
# --------------------------------------------------


def download_video(
    url: str,
    filename: Optional[str] = None,
) -> Dict[str, Any]:

    log_section("Video Download")
    log_kv("Source URL", url)

    # --------------------------------------------------
    # FETCH METADATA ONLY
    # --------------------------------------------------

    ydl_meta_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    # 🔇 pyright false-positive (yt-dlp typing bug)
    with yt_dlp.YoutubeDL(ydl_meta_opts) as ydl:  # pyright: ignore[reportArgumentType]
        info = ydl.extract_info(url, download=False)

    video_id = str(info.get("id") or "")
    title = str(info.get("title") or "unknown")
    source = str(info.get("extractor_key") or "unknown")

    if not video_id:
        raise RuntimeError("Could not extract video ID")

    log_kv("Video ID", video_id)
    log_kv("Title", title)
    log_kv("Platform", source)

    _validate_video_info(info)

    # --------------------------------------------------
    # CACHE CHECK
    # --------------------------------------------------

    cached = _video_exists(video_id)
    if cached:
        log.info("Video already downloaded, using cached file")
        return {
            "video_id": video_id,
            "title": title,
            "source": source,
            "path": cached,
            "duration": info.get("duration"),
            "resolution": info.get("height"),
        }

    # --------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------

    output_template = str(RAW_VIDEO_DIR / "%(id)s.%(ext)s")

    ydl_download_opts = {
        "outtmpl": output_template,
        "format": "bv*[ext=mp4]/b[ext=mp4]/bv*/b",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }

    log.info("Downloading video…")

    # 🔇 pyright false-positive again
    with yt_dlp.YoutubeDL(
        ydl_download_opts
    ) as ydl:  # pyright: ignore[reportArgumentType]
        result = ydl.extract_info(url, download=True)
    final_path = Path(ydl.prepare_filename(result))

    # yt-dlp may download webm first, then merge to mp4
    if final_path.suffix != ".mp4":
        mp4_path = final_path.with_suffix(".mp4")
        if mp4_path.exists():
            final_path = mp4_path

    if not final_path.exists():
        raise RuntimeError(f"Download failed — file not found at {final_path}")

    log.info(f"Download complete → {final_path}")

    return {
        "video_id": video_id,
        "title": title,
        "source": source,
        "path": final_path,  # ✅ FIXED
        "duration": info.get("duration"),
        "resolution": info.get("height"),
    }
