from __future__ import annotations

from pathlib import Path
from typing import List, Dict

from moviepy.editor import (
    VideoFileClip,
    CompositeVideoClip,
    TextClip,
)

from src.config.settings import VIDEO_CODEC, VIDEO_BITRATE, TEXT_FONT
from src.config.paths import TAGGED_VIDEO_DIR
from src.utils.logger import get_logger, log_section

log = get_logger("overlay")


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def _build_label(item: Dict) -> str:
    """
    Create human-friendly overlay text.
    """
    emoji_map = {
        "watch": "⌚",
        "shoe": "👟",
        "necklace": "💎",
        "ring": "💍",
        "bracelet": "📿",
    }

    emoji = emoji_map.get(item["item"], "")
    return f"{emoji} {item['item'].title()}: {item['price_range']}"


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def render_overlay(
    video_path: Path,
    video_id: str,
    priced_items: List[Dict],
) -> Path:
    """
    Render price overlays on top of the video.

    Returns:
        Path to tagged video
    """

    log_section("Rendering Video Overlay")

    if not priced_items:
        log.warning("No priced items found — skipping overlay")
        return video_path

    clip = VideoFileClip(str(video_path))
    overlays = []

    margin_x = 40
    margin_y = 120
    line_spacing = 58

    for idx, item in enumerate(priced_items):
        label = _build_label(item)

        txt = (
            TextClip(
                label,
                fontsize=42,
                font=TEXT_FONT,
                color="white",
                stroke_color="black",
                stroke_width=2,
                method="label",
            )
            .set_position(
                ("left", margin_y + idx * line_spacing)
            )
            .set_duration(clip.duration)
        )

        overlays.append(txt)

    final = CompositeVideoClip([clip, *overlays])

    output_path = TAGGED_VIDEO_DIR / f"{video_id}_tagged.mp4"

    log.info("Exporting final video...")

    final.write_videofile(
        str(output_path),
        codec=VIDEO_CODEC,
        bitrate=VIDEO_BITRATE,
        audio_codec="aac",
        threads=4,
        logger=None,
    )

    clip.close()
    final.close()

    log.info(f"Tagged video saved → {output_path}")

    return output_path
