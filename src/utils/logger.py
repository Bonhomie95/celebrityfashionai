from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TaskID,
)

console = Console()


# --------------------------------------------------
# LOGGER
# --------------------------------------------------


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # prevent duplicate handlers

    logger.setLevel(level)

    handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
    )

    formatter = logging.Formatter(
        "%(name)s | %(message)s",
        datefmt="[%H:%M:%S]",
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# --------------------------------------------------
# PROGRESS TRACKER
# --------------------------------------------------


class ProgressTracker:
    """
    Simple progress bar wrapper.
    Designed for:
    - frame extraction
    - detection loops
    - batch processing
    """

    def __init__(self, title: str, total: int):
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        )
        self.task_id: Optional[TaskID] = None
        self.title = title
        self.total = total

    def __enter__(self):
        self.progress.start()
        self.task_id = self.progress.add_task(
            self.title,
            total=self.total,
        )
        return self

    def advance(self, step: int = 1):
        if self.task_id is not None:
            self.progress.advance(self.task_id, step)

    def update(self, completed: int):
        if self.task_id is not None:
            self.progress.update(self.task_id, completed=completed)

    def __exit__(self, exc_type, exc, tb):
        self.progress.stop()


# --------------------------------------------------
# HELPER
# --------------------------------------------------


def log_section(title: str) -> None:
    console.rule(f"[bold cyan]{title}")


def log_kv(key: str, value: object) -> None:
    console.print(f"[bold]{key}[/]: {value}")
