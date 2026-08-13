"""
MetaDrive backend driver -- now a plain bridge to the shared module.

The implementation used to live here. It now lives in
`scenarios/common/driver.py`, because the same control law drives on Udacity
too and two copies would make false the assumption the comparison rests on --
the same controller on different simulators. Bit-for-bit equivalence was
verified before moving it (0.0 deviation over 300 steps).

Only the re-exports remain, so existing imports keep working. Do not add logic
here.
"""
from __future__ import annotations

from scenarios.common.driver import Driver, LateralFeedbackDriver, _clip

__all__ = ["Driver", "LateralFeedbackDriver", "_clip"]
