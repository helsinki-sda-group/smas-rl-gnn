from __future__ import annotations

from typing import Any, Dict, Optional


_LATEST_POLICY_TIMING: Optional[Dict[str, Any]] = None


def set_latest_policy_timing(payload: Dict[str, Any]) -> None:
    global _LATEST_POLICY_TIMING
    _LATEST_POLICY_TIMING = dict(payload)


def pop_latest_policy_timing() -> Optional[Dict[str, Any]]:
    global _LATEST_POLICY_TIMING
    out = _LATEST_POLICY_TIMING
    _LATEST_POLICY_TIMING = None
    return out
