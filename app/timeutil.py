"""
timeutil.py
-----------
ISO-8601 타임스탬프 간 소요시간(초) 계산 공용 헬퍼.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional


def iso_duration_sec(
    started: Optional[str], finished: Optional[str]
) -> Optional[float]:
    """started~finished 소요시간(초). 둘 중 하나라도 없으면 None."""
    if not started or not finished:
        return None
    try:
        s = datetime.fromisoformat(started)
        f = datetime.fromisoformat(finished)
    except ValueError:
        return None
    return round((f - s).total_seconds(), 3)
