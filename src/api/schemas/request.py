"""
Pydantic request models for API validation.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ProfileAnalysisRequest(BaseModel):
    """Request model for complete profile analysis"""
    bio_text: str = Field(..., min_length=10, max_length=1000, description="Profile bio text")
    profile_id: Optional[str] = Field(None, max_length=256, description="Optional profile identifier")

    @field_validator('bio_text')
    @classmethod
    def strip_bio(cls, v: str) -> str:
        # min_length already rejects empty/whitespace once stripped; this just normalizes.
        return v.strip()


class TextAnalysisRequest(BaseModel):
    """Request model for text-only analysis"""
    text: str = Field(..., min_length=10, max_length=1000, description="Text to analyze")

    @field_validator('text')
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()
