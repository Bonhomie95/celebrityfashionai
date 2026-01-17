from __future__ import annotations

from pathlib import Path
from typing import Literal

import clip
import torch
from typing import cast

from PIL import Image

from src.utils.logger import get_logger

log = get_logger("glasses-classifier")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)

TEXT_PROMPTS = [
    "a person wearing glasses",
    "a person wearing sunglasses",
    "a person without glasses",
]


def classify_glasses(image_path: Path) -> Literal["glasses", "no_glasses"]:
    img_tensor = PREPROCESS(Image.open(image_path).convert("RGB"))
    img_tensor = cast(torch.Tensor, img_tensor)

    image = img_tensor.unsqueeze(0).to(DEVICE)

    text = clip.tokenize(TEXT_PROMPTS).to(DEVICE)

    with torch.no_grad():
        image_features = MODEL.encode_image(image)
        text_features = MODEL.encode_text(text)

        logits = (image_features @ text_features.T).softmax(dim=-1)
        probs = logits[0].cpu().tolist()

    glasses_score = probs[0] + probs[1]
    no_glasses_score = probs[2]

    result = "glasses" if glasses_score > no_glasses_score else "no_glasses"

    log.info(f"Glasses detection → {result} ({glasses_score:.2f})")

    return result
