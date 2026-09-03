"""
MetaDrive backend driver: re-exports of the shared implementation.

The control law lives in `scenarios/common/driver.py`, which both backends
import. This module only re-exports `Driver`, `LateralFeedbackDriver` and `_clip` from
there, so imports written against this path keep working.
"""
from __future__ import annotations

from scenarios.common.driver import Driver, LateralFeedbackDriver, _clip

__all__ = ["Driver", "LateralFeedbackDriver", "_clip"]
