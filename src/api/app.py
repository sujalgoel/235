"""
FastAPI application for RealityCheck AI.
"""

import time
from typing import Optional, Any, Dict
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile
import os
import numpy as np

from src.core.pipeline import AnalysisPipeline
from src.api.schemas.request import TextAnalysisRequest
from src.api.schemas.response import (
    ProfileAnalysisResponse,
    ImageAnalysisResponse,
    TextAnalysisResponse,
    HealthResponse,
    ErrorResponse
)
from src.utils.logging import configure_logging, get_logger
from src.utils.exceptions import RealityCheckException
from config.base import get_config


async def _read_upload_with_limit(upload: UploadFile, max_bytes: int) -> bytes:
    """Read an UploadFile in chunks, rejecting early if it exceeds max_bytes.

    Reading the whole file before checking size lets a client OOM the worker
    by sending a multi-GB body. Streaming and checking the running total
    bounds memory at max_bytes + chunk_size.
    """
    chunk_size = 1024 * 1024  # 1 MiB
    buffer = bytearray()
    while True:
        chunk = await upload.read(chunk_size)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large (max: {max_bytes // (1024 * 1024)}MB)"
            )
    return bytes(buffer)

# Configuration
ENV = os.getenv("ENVIRONMENT", "development")
config = get_config(ENV)

# Configure logging
configure_logging(
    log_level=config.LOGGING.log_level,
    log_format=config.LOGGING.log_format,
    log_file=config.LOGGING.log_file
)

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RealityCheck AI API",
    version="1.0.0",
    description="Multimodal fake profile detection with explainable AI",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.API.cors_origins,
    allow_credentials=config.API.cors_allow_credentials,
    allow_methods=config.API.cors_allow_methods,
    allow_headers=config.API.cors_allow_headers,
)

# Initialize pipeline (singleton)
pipeline = AnalysisPipeline(config)


def make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert non-JSON-serializable objects to serializable types.

    This function is critical for API responses because AI models return:
    - NumPy arrays (not JSON-serializable)
    - PyTorch tensors (not JSON-serializable)
    - NumPy scalars (int64, float32, etc.)

    The function walks through nested data structures and converts everything
    to native Python types that can be serialized to JSON.

    Handles:
    - numpy arrays → Python lists
    - numpy scalars → Python int/float
    - torch tensors → Python lists
    - nested dicts/lists (recursive)

    Args:
        obj: Any object that might contain non-serializable types

    Returns:
        JSON-serializable version of the object
    """
    # Handle nested dictionaries recursively
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    # Handle lists and tuples recursively
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    # Convert NumPy arrays to Python lists
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Convert NumPy scalars to Python primitives
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    # Convert PyTorch tensors to Python lists
    # Must move to CPU first if on GPU, then detach from computation graph
    elif hasattr(obj, 'detach'):  # torch tensor
        return obj.detach().cpu().numpy().tolist()
    # Native Python types are already serializable
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # Fallback: convert unknown types to string representation
        # This prevents crashes but may lose some information
        return str(obj)



@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    logger.info("api_startup", environment=ENV)
    try:
        # Initialize AI pipeline
        pipeline.initialize()
        logger.info("pipeline_ready", message="Manual upload mode - Image + Text analysis")

    except Exception as e:
        logger.error("startup_failed", error=str(e))
        raise


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "service": "RealityCheck AI",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for load balancers.

    Returns service status and module readiness.
    """
    status_info = pipeline.get_status()

    return HealthResponse(
        status="healthy" if status_info["initialized"] else "initializing",
        version="1.0.0",
        modules=status_info["modules"]
    )


@app.post("/api/v1/analyze/profile", tags=["Analysis"])
async def analyze_profile(
    image: UploadFile = File(..., description="Profile image"),
    bio_text: str = Form(..., min_length=10, max_length=1000, description="Profile bio text"),
    profile_id: Optional[str] = Form(None, max_length=256, description="Optional profile ID")
):
    """
    Analyze image and text for AI-generated content.

    Returns comprehensive analysis including:
    - Image authenticity (Ensemble: EfficientNet-B7 + XceptionNet + CLIP)
    - Text authenticity (Ensemble: OpenAI Detector + ChatGPT Detector + Rules)
    - Final trust score with detailed explanations
    """
    try:
        # Log incoming analysis request for monitoring
        logger.info("profile_analysis_request",
                   profile_id=profile_id,
                   filename=image.filename)

        # Step 1: Stream upload and reject early if it exceeds the size cap
        content = await _read_upload_with_limit(image, config.API.max_upload_size)

        # Step 2: Save uploaded image to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image.filename).suffix) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # Step 3: Run AI analysis pipeline
            # This processes both image and text through ensemble models
            # Returns comprehensive analysis with trust score
            result = pipeline.analyze_profile(
                image_path=tmp_path,      # Path to temporary image file
                text=bio_text,             # User-provided bio text
                profile_id=profile_id      # Optional identifier for logging
            )

            # Step 4: Convert numpy/torch objects to JSON-serializable types
            # AI models return numpy arrays and tensors that can't be directly serialized
            serializable_result = make_json_serializable(result)

            # Step 5: Flatten nested trust score data for easier frontend consumption
            # Move trust_score_result fields to top level for cleaner API response
            if "trust_score_result" in serializable_result:
                trust_data = serializable_result.pop("trust_score_result")
                # Extract main trust score metrics
                serializable_result["final_trust_score"] = trust_data.get("trust_score", 0.0)
                serializable_result["interpretation"] = trust_data.get("interpretation", "")
                serializable_result["trust_level"] = trust_data.get("trust_level", "")
                # Preserve detailed analysis for frontend display
                serializable_result["trust_score_details"] = {
                    "module_scores": trust_data.get("module_scores", {}),
                    "contributing_factors": trust_data.get("contributing_factors", {})
                }

            return JSONResponse(content=serializable_result)

        finally:
            # Always clean up temporary file. OSError covers both permission
            # issues and FileNotFoundError; we log and move on so cleanup
            # failures never propagate to the client.
            try:
                os.unlink(tmp_path)
            except OSError as cleanup_err:
                logger.warning("temp_file_cleanup_failed", path=tmp_path, error=str(cleanup_err))

    except HTTPException:
        raise  # Preserve HTTP errors (e.g. 413) we raised intentionally
    except RealityCheckException as e:
        logger.error("analysis_failed", error=str(e), error_type=type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed"
        )
    except Exception as e:
        logger.error("unexpected_error", error=str(e), error_type=type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.post("/api/v1/analyze/image", tags=["Analysis"])
async def analyze_image(image: UploadFile = File(..., description="Image to analyze")):
    """
    Analyze image only (no text analysis).

    Returns:
    - Face detection results
    - Fake/real classification
    - Grad-CAM visualization
    """
    try:
        logger.info("image_analysis_request", filename=image.filename)

        # Validate and save image
        content = await _read_upload_with_limit(image, config.API.max_upload_size)

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image.filename).suffix) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            result = pipeline.analyze_image_only(tmp_path)
            serializable_result = make_json_serializable(result)
            return JSONResponse(content=serializable_result)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError as cleanup_err:
                logger.warning("temp_file_cleanup_failed", path=tmp_path, error=str(cleanup_err))

    except HTTPException:
        raise
    except Exception as e:
        logger.error("image_analysis_failed", error=str(e), error_type=type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Image analysis failed"
        )


@app.post("/api/v1/analyze/text", response_model=TextAnalysisResponse, tags=["Analysis"])
async def analyze_text(text: str = Form(..., min_length=10, max_length=1000, description="Text to analyze")):
    """
    Analyze text only (no image analysis).

    Returns:
    - Human vs AI classification
    - SHAP/LIME token importance
    - Linguistic pattern analysis
    """
    try:
        logger.info("text_analysis_request", length=len(text))

        result = pipeline.analyze_text_only(text)
        if "text_analysis" not in result:
            logger.error("pipeline_missing_text_analysis", keys=list(result.keys()))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Text analysis failed"
            )
        serializable_result = make_json_serializable(result["text_analysis"])
        return JSONResponse(content=serializable_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("text_analysis_failed", error=str(e), error_type=type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Text analysis failed"
        )


@app.exception_handler(RealityCheckException)
async def reality_check_exception_handler(request, exc: RealityCheckException):
    """Handle custom exceptions"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message=str(exc),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error("unhandled_exception", error=str(exc), type=type(exc).__name__)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.app:app",
        host=config.API.host,
        port=config.API.port,
        reload=config.API.reload
    )
