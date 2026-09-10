"""Image parser placeholder after text extraction support was removed."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.core.logging import get_logger

log = get_logger(__name__)


def parse(filepath: str | Path) -> List[Tuple[str, Dict[str, Any]]]:
    filepath = Path(filepath)
    log.warning("Image file %s is not indexed because direct text extraction support was removed.", filepath.name)
    return []
