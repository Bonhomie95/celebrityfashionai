from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Any, Mapping, cast

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

    # ✅ positional dict — correct yt-dlp usage
    with yt_dlp.YoutubeDL(cast(Any, ydl_meta_opts)) as ydl:
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

    output_template = filename or f"{video_id}.%(ext)s"

    ydl_download_opts = {
        "outtmpl": str(RAW_VIDEO_DIR / output_template),
        "format": "bestvideo[ext=mp4]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }

    log.info("Downloading video…")

    # ✅ positional dict again
    with yt_dlp.YoutubeDL(cast(Any, ydl_meta_opts)) as ydl:
        ydl.download([url])

    downloaded = _video_exists(video_id)
    if not downloaded:
        raise RuntimeError("Download failed — file not found")

    log.info("Download complete")

    return {
        "video_id": video_id,
        "title": title,
        "source": source,
        "path": downloaded,
        "duration": info.get("duration"),
        "resolution": info.get("height"),
    }
