from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip

from src.config.paths import OUTPUT_DIR
from src.utils.logger import get_logger, log_section
import numpy as np

log = get_logger("overlay")


def _build_label(item: Dict[str, Any]) -> str:
    price = item.get("price_range", "Unknown price")
    name = item.get("item", "Item")
    return f"{name} • {price}"


def _render_text_image(
    text: str,
    width: int,
    font_size: int = 42,
) -> Image.Image:
    """
    Render text into a PIL image (no ImageMagick).
    """
    padding = 20
    bg_color = (0, 0, 0, 160)  # semi-transparent
    text_color = (255, 255, 255, 255)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    dummy = Image.new("RGBA", (width, 10))
    draw = ImageDraw.Draw(dummy)
    text_bbox = draw.multiline_textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    img = Image.new(
        "RGBA",
        (
            int(text_w + padding * 2),
            int(text_h + padding * 2),  # ✅ must be int
        ),
        bg_color,
    )

    draw = ImageDraw.Draw(img)
    draw.multiline_text(
        (padding, padding),
        text,
        fill=text_color,
        font=font,
        align="center",
    )

    return img


def render_overlay(
    video_path: Path,
    video_id: str,
    priced_items: List[Dict[str, Any]],
) -> Path:
    """
    Render price overlays using PIL (Windows-safe).
    """

    log_section("Rendering Video Overlay")

    video = VideoFileClip(str(video_path))
    video_w, video_h = video.size

    overlays: List[ImageClip] = []

    for idx, item in enumerate(priced_items[:5]):  # limit clutter
        label = _build_label(item)

        img = _render_text_image(
            text=label,
            width=int(video_w * 0.9),
        )

        clip = (
            ImageClip(np.array(img))
            .set_duration(video.duration)
            .set_position(
                (
                    "center",
                    int(video_h * 0.1 + idx * 70),
                )
            )
        )

        overlays.append(clip)

    final = CompositeVideoClip([video] + overlays)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{video_id}_priced.mp4"

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
