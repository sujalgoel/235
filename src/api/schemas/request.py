"""
Pydantic request models for API validation.
"""

from typing import Optional
from pydantic import BaseModel, Field, validator


class ProfileAnalysisRequest(BaseModel):
    """Request model for complete profile analysis"""
    bio_text: str = Field(..., min_length=10, max_length=1000, description="Profile bio text")
    profile_id: Optional[str] = Field(None, description="Optional profile identifier")

    @validator('bio_text')
    def validate_bio(cls, v):
        if not v or not v.strip():
            raise ValueError("Bio text cannot be empty")
        return v.strip()


class TextAnalysisRequest(BaseModel):
    """Request model for text-only analysis"""
    text: str = Field(..., min_length=10, max_length=1000, description="Text to analyze")

    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()
