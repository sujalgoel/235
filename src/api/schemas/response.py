"""Pydantic response models exposed in the OpenAPI schema.

Only the response models referenced by FastAPI route decorators or returned
to clients are kept here. Detector outputs flow through ad-hoc dicts shaped
by `_shape_response` in src/api/app.py.
"""

from typing import Dict

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    modules: Dict[str, bool] = Field(default_factory=dict, description="Per-module readiness")


class ErrorResponse(BaseModel):
    """Shape of every error response from the API."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable explanation")
    timestamp: str = Field(..., description="UTC ISO-8601 timestamp")
