from __future__ import annotations

from typing import Any, Optional

import numpy as np


_LATEST_POLICY_STEP: Optional[dict[str, Any]] = None


def set_latest_policy_step(logits_k: Any, mask_k: Any) -> None:
    global _LATEST_POLICY_STEP
    logits_np = np.asarray(logits_k.detach().cpu().numpy(), dtype=np.float32)
    mask_np = np.asarray(mask_k.detach().cpu().numpy(), dtype=bool)
    _LATEST_POLICY_STEP = {
        "logits_k": logits_np.copy(),
        "mask_k": mask_np.copy(),
    }


def pop_latest_policy_step() -> Optional[dict[str, np.ndarray]]:
    global _LATEST_POLICY_STEP
    payload = _LATEST_POLICY_STEP
    _LATEST_POLICY_STEP = None
    return payload
