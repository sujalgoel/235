"""
Pydantic response models for API.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ImageAnalysisResponse(BaseModel):
    """Image analysis results"""
    score: float = Field(..., ge=0, le=1, description="Authenticity score")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    prediction: str = Field(..., description="real or fake")
    explanation: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TextAnalysisResponse(BaseModel):
    """Text analysis results"""
    score: float = Field(..., ge=0, le=1, description="Authenticity score")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    prediction: str = Field(..., description="human or ai")
    explanation: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetadataAnalysisResponse(BaseModel):
    """Metadata analysis results"""
    score: float = Field(..., ge=0, le=1, description="Authenticity score")
    confidence: float = Field(..., ge=0, le=1, description="Analysis confidence")
    prediction: str = Field(..., description="real or fake")
    explanation: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TrustScoreResponse(BaseModel):
    """Trust score results"""
    trust_score: float = Field(..., ge=0, le=1, description="Final trust score")
    trust_level: str = Field(..., description="Trust level category")
    interpretation: str = Field(..., description="Human-readable interpretation")
    module_scores: Dict[str, float] = Field(default_factory=dict)
    module_confidences: Dict[str, float] = Field(default_factory=dict)
    contributing_factors: Dict[str, List[str]] = Field(default_factory=dict)


class ProfileAnalysisResponse(BaseModel):
    """Complete profile analysis response"""
    profile_id: str = Field(..., description="Profile identifier")
    timestamp: str = Field(..., description="Analysis timestamp")
    image_analysis: Optional[ImageAnalysisResponse] = None
    text_analysis: Optional[TextAnalysisResponse] = None
    metadata_analysis: Optional[MetadataAnalysisResponse] = None
    trust_score_result: TrustScoreResponse
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    modules: Dict[str, bool] = Field(default_factory=dict, description="Module status")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: str = Field(..., description="Error timestamp")
