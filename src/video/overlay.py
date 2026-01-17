from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip

from src.config.paths import TAGGED_VIDEO_DIR
from src.utils.logger import get_logger, log_section
from PIL import ImageFont
from typing import Union

log = get_logger("overlay")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

ITEM_DISPLAY_SECONDS = 1.8
OVERLAY_BG_COLOR = (0, 0, 0, 180)
TEXT_COLOR = (255, 255, 255, 255)
LINE_COLOR = (255, 255, 255, 220)
PADDING = 16
FONT_SIZE = 40


# --------------------------------------------------
# HELPERS
# --------------------------------------------------


def _load_font(size: int) -> Union[ImageFont.ImageFont, ImageFont.FreeTypeFont]:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _price_label(item: Dict[str, Any]) -> str:
    return f"{item.get('price_range', 'Unknown price')}"


def _bbox_center(bbox: List[int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2


def _render_overlay_frame(
    video_size: Tuple[int, int],
    bbox: List[int],
    label: str,
) -> Image.Image:
    """
    Create a single RGBA overlay frame:
    - leader line
    - price box
    """
    video_w, video_h = video_size

    img = Image.new("RGBA", (video_w, video_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_font(FONT_SIZE)

    # --------------------------------------------------
    # Overlay box geometry
    # --------------------------------------------------

    overlay_x = int(video_w * 0.65)
    overlay_y = int(video_h * 0.2)

    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    box_w = text_w + PADDING * 2
    box_h = text_h + PADDING * 2

    box_rect = [
        overlay_x,
        overlay_y,
        overlay_x + box_w,
        overlay_y + box_h,
    ]

    # --------------------------------------------------
    # Draw box
    # --------------------------------------------------

    draw.rectangle(box_rect, fill=OVERLAY_BG_COLOR)

    draw.text(
        (overlay_x + PADDING, overlay_y + PADDING),
        label,
        fill=TEXT_COLOR,
        font=font,
    )

    # --------------------------------------------------
    # Leader line
    # --------------------------------------------------

    item_center = _bbox_center(bbox)
    line_target = (overlay_x, overlay_y + box_h // 2)

    draw.line(
        [item_center, line_target],
        fill=LINE_COLOR,
        width=4,
    )

    # --------------------------------------------------
    # Highlight bbox (subtle)
    # --------------------------------------------------

    x1, y1, x2, y2 = bbox
    draw.rectangle(
        [x1, y1, x2, y2],
        outline=(255, 255, 255, 220),
        width=3,
    )

    return img


# --------------------------------------------------
# MAIN
# --------------------------------------------------


def render_overlay(
    video_path: Path,
    video_id: str,
    priced_items: List[Dict[str, Any]],
) -> Path:
    """
    Render sequential pricing overlays:
    one item at a time, with leader line.
    """

    log_section("Rendering Video Overlay")

    video = VideoFileClip(str(video_path))
    video_w, video_h = video.size

    clips: List[Any] = [video]

    for idx, item in enumerate(priced_items):
        start_t = idx * ITEM_DISPLAY_SECONDS
        end_t = start_t + ITEM_DISPLAY_SECONDS

        bbox = item["bbox"]
        label = _price_label(item)

        overlay_img = _render_overlay_frame(
            video_size=(video_w, video_h),
            bbox=bbox,
            label=label,
        )

        overlay_clip = (
            ImageClip(np.array(overlay_img))
            .set_start(start_t)
            .set_end(end_t)
            .set_position((0, 0))
        )

        clips.append(overlay_clip)

    final = CompositeVideoClip(clips, size=video.size)

    TAGGED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TAGGED_VIDEO_DIR / f"{video_id}_priced.mp4"

    final.write_videofile(
        str(out_path),
        codec="libx264",
        audio_codec="aac",
        fps=video.fps,
        threads=4,
        logger=None,
    )

    video.close()
    final.close()

    log.info(f"Overlay video saved → {out_path}")
    return out_path
