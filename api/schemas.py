"""Pydantic models for API request/response validation."""

from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    n_samples: int = Field(default=500, ge=10, le=5000)
    seed: int = Field(default=42)
    rare_fraction: float = Field(default=0.05, gt=0, lt=1)
    # Optional: override scenario default bounds for LHS sampling
    param_lower: Optional[list[float]] = Field(default=None)
    param_upper: Optional[list[float]] = Field(default=None)


class ScenarioInfo(BaseModel):
    name: str
    description: str
    param_names: list[str]
    param_lower: list[float]
    param_upper: list[float]


class StatusResponse(BaseModel):
    status: Literal["queued", "running", "done", "error"]
    error: Optional[str] = None
    # populated only when status == "done"
    scenario_name: Optional[str] = None
    n_samples: Optional[int] = None
    failure_rate: Optional[float] = None
    rare_failure_rate: Optional[float] = None
    pod_n_modes: Optional[int] = None
    param_names: Optional[list[str]] = None
    trajectories: Optional[list] = None          # (N, T, D) as nested lists
    safe_mask: Optional[list[bool]] = None       # (N,)
    rare_failure_idx: Optional[list[int]] = None
    nominal_trajectory: Optional[list] = None    # (1, T, D)
    rare_params_summary: Optional[list[dict]] = None
    safety_margins: Optional[list[float]] = None


class ExplainResponse(BaseModel):
    summary: str
