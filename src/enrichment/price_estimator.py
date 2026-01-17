from __future__ import annotations

from typing import Dict, List
from pathlib import Path
import json

from src.config.settings import (
    DEFAULT_PRICE_RANGE,
    PRICE_CONFIDENCE_MIN,
    ENABLE_WEB_LOOKUP,
)
from src.config.paths import MODEL_DIR
from src.utils.logger import get_logger, log_section

log = get_logger("price-estimator")


# --------------------------------------------------
# LOAD HEURISTIC RULES
# --------------------------------------------------

BRAND_RULES_PATH = MODEL_DIR / "brand_rules.json"

if BRAND_RULES_PATH.exists():
    with open(BRAND_RULES_PATH, "r", encoding="utf-8") as f:
        BRAND_RULES = json.load(f)
else:
    # Fallback rules (safe defaults)
    BRAND_RULES = {
        "watch": {
            "luxury_range": "$15,000 – $300,000",
            "regular_range": "$500 – $5,000",
            "luxury_keywords": ["rolex", "patek", "audemars", "richard mille"],
        },
        "shoe": {
            "luxury_range": "$1,000 – $10,000",
            "regular_range": "$150 – $800",
            "luxury_keywords": ["louboutin", "gucci", "balenciaga"],
        },
        "necklace": {
            "luxury_range": "$5,000 – $250,000",
            "regular_range": "$200 – $3,000",
            "luxury_keywords": ["cartier", "tiffany", "bvlgari"],
        },
        "ring": {
            "luxury_range": "$8,000 – $500,000",
            "regular_range": "$300 – $4,000",
            "luxury_keywords": ["cartier", "harry winston"],
        },
        "bracelet": {
            "luxury_range": "$3,000 – $120,000",
            "regular_range": "$150 – $2,500",
            "luxury_keywords": ["cartier", "van cleef"],
        },
    }


# --------------------------------------------------
# CORE LOGIC
# --------------------------------------------------


def _estimate_price_for_item(
    item_type: str,
    confidence: float,
) -> Dict:
    """
    Heuristic-based price estimation.
    """

    rules = BRAND_RULES.get(item_type.lower())
    if not rules:
        return {
            "price_range": DEFAULT_PRICE_RANGE,
            "luxury": False,
            "reason": "no_rules_for_item",
        }

    # Confidence-driven assumption:
    # High confidence detection → likely prominent/luxury
    is_luxury = confidence >= PRICE_CONFIDENCE_MIN

    price_range = rules["luxury_range"] if is_luxury else rules["regular_range"]

    return {
        "price_range": price_range,
        "luxury": is_luxury,
        "reason": (
            "high_confidence_detection" if is_luxury else "low_confidence_detection"
        ),
    }


# --------------------------------------------------
# PUBLIC API
# --------------------------------------------------


def estimate_prices(
    quality_items: List[Dict],
) -> List[Dict]:
    """
    Attach price estimates to high-quality cropped items.

    Returns:
        [
          {
            "item": str,
            "confidence": float,
            "crop_path": Path,
            "price_range": str,
            "luxury": bool,
            "estimation_reason": str
          }
        ]
    """

    log_section("Price Estimation")

    enriched: List[Dict] = []

    for item in quality_items:
        item_type = item["item"]
        confidence = item["confidence"]

        estimate = _estimate_price_for_item(
            item_type=item_type,
            confidence=confidence,
        )

        enriched_item = {
            **item,
            "price_range": estimate["price_range"],
            "luxury": estimate["luxury"],
            "estimation_reason": estimate["reason"],
        }

        enriched.append(enriched_item)

    log.info(f"Estimated prices for {len(enriched)} items")

    if ENABLE_WEB_LOOKUP:
        log.warning(
            "ENABLE_WEB_LOOKUP is true, " "but web lookup is not implemented yet."
        )

    return enriched
