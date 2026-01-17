from __future__ import annotations

import sys
from typing import Optional

from src.pipeline.orchestrator import run
from src.utils.logger import get_logger, log_section

log = get_logger("main")


def main(url: Optional[str] = None) -> None:
    """
    Entry point for the celebrity fashion pipeline.
    """

    log_section("CELEBRITY FASHION AI")

    if not url:
        if len(sys.argv) < 2:
            log.error("Usage: python -m src <video_url>")
            sys.exit(1)
        url = sys.argv[1]

    try:
        result = run(url)
        if result:
            log.info(f"Pipeline finished successfully → {result}")
        else:
            log.warning("Pipeline finished with no output")
    except Exception as e:
        log.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
