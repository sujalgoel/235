"""FastAPI application for RealityCheck AI."""

import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.base import get_config
from src.api.schemas.response import ErrorResponse, HealthResponse
from src.core.pipeline import AnalysisPipeline
from src.utils.exceptions import RealityCheckException
from src.utils.logging import configure_logging, get_logger


ENV = os.getenv("ENVIRONMENT", "development")
config = get_config(ENV)

configure_logging(
    log_level=config.LOGGING.log_level,
    log_format=config.LOGGING.log_format,
    log_file=config.LOGGING.log_file,
)
logger = get_logger(__name__)

# Singleton pipeline. Models are loaded on app startup, not at import time,
# so the test suite can mock this object before any heavy ML import runs.
pipeline = AnalysisPipeline(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("api_startup", environment=ENV)
    try:
        pipeline.initialize()
        logger.info("pipeline_ready")
    except Exception as e:
        logger.error("startup_failed", error=str(e))
        raise
    yield


# Hide interactive docs in production so we don't broadcast our schema.
_docs_enabled = ENV != "production"
app = FastAPI(
    title="RealityCheck AI API",
    version="1.0.0",
    description="Multimodal fake profile detection with explainable AI",
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
    openapi_url="/openapi.json" if _docs_enabled else None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.API.cors_origins,
    allow_credentials=config.API.cors_allow_credentials,
    allow_methods=config.API.cors_allow_methods,
    allow_headers=config.API.cors_allow_headers,
)


def make_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy / torch types into JSON-friendly equivalents."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if torch.is_tensor(obj):
        # torch.is_tensor avoids false positives on transformers BatchEncoding
        # which also exposes a `.detach` attribute.
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def _shape_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten trust_score_result onto the response root so every endpoint
    emits the same envelope."""
    serializable = make_json_serializable(result)
    if "trust_score_result" in serializable:
        trust_data = serializable.pop("trust_score_result")
        serializable["final_trust_score"] = trust_data.get("trust_score", 0.0)
        serializable["interpretation"] = trust_data.get("interpretation", "")
        serializable["trust_level"] = trust_data.get("trust_level", "")
        serializable["trust_score_details"] = {
            "module_scores": trust_data.get("module_scores", {}),
            "contributing_factors": trust_data.get("contributing_factors", {}),
        }
    return serializable


async def _read_upload_with_limit(upload: UploadFile, max_bytes: int) -> bytes:
    """Stream an UploadFile in 1 MiB chunks; reject early if it exceeds max_bytes.

    Reading the whole body before checking size lets a client OOM the worker
    by sending a multi-GB upload. Streaming bounds memory at max_bytes + 1 MiB.
    """
    chunk_size = 1024 * 1024
    buffer = bytearray()
    while True:
        chunk = await upload.read(chunk_size)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large (max: {max_bytes // (1024 * 1024)}MB)",
            )
    return bytes(buffer)


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "RealityCheck AI",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs" if _docs_enabled else None,
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    status_info = pipeline.get_status()
    return HealthResponse(
        status="healthy" if status_info["initialized"] else "initializing",
        version="1.0.0",
        modules=status_info["modules"],
    )


@app.post("/api/v1/analyze/profile", tags=["Analysis"])
async def analyze_profile(
    image: UploadFile = File(..., description="Profile image"),
    bio_text: str = Form(..., min_length=10, max_length=1000, description="Profile bio text"),
    profile_id: Optional[str] = Form(None, max_length=256, description="Optional profile ID"),
):
    """Analyze image + text. Returns the unified envelope."""
    try:
        logger.info("profile_analysis_request", profile_id=profile_id, filename=image.filename)
        content = await _read_upload_with_limit(image, config.API.max_upload_size)

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image.filename).suffix) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            result = pipeline.analyze_profile(
                image_path=tmp_path,
                text=bio_text,
                profile_id=profile_id,
            )
            return JSONResponse(content=_shape_response(result))
        finally:
            try:
                os.unlink(tmp_path)
            except OSError as cleanup_err:
                logger.warning("temp_file_cleanup_failed", path=tmp_path, error=str(cleanup_err))

    except HTTPException:
        raise
    except RealityCheckException as e:
        logger.error("analysis_failed", error=str(e), error_type=type(e).__name__)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Analysis failed")
    except Exception as e:
        logger.error("unexpected_error", error=str(e), error_type=type(e).__name__)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@app.post("/api/v1/analyze/image", tags=["Analysis"])
async def analyze_image(image: UploadFile = File(..., description="Image to analyze")):
    """Analyze image only. Same envelope as /analyze/profile with text_analysis = null."""
    try:
        logger.info("image_analysis_request", filename=image.filename)
        content = await _read_upload_with_limit(image, config.API.max_upload_size)

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image.filename).suffix) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            result = pipeline.analyze_image_only(tmp_path)
            return JSONResponse(content=_shape_response(result))
        finally:
            try:
                os.unlink(tmp_path)
            except OSError as cleanup_err:
                logger.warning("temp_file_cleanup_failed", path=tmp_path, error=str(cleanup_err))

    except HTTPException:
        raise
    except Exception as e:
        logger.error("image_analysis_failed", error=str(e), error_type=type(e).__name__)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Image analysis failed")


@app.post("/api/v1/analyze/text", tags=["Analysis"])
async def analyze_text(
    text: str = Form(..., min_length=10, max_length=1000, description="Text to analyze"),
):
    """Analyze text only. Same envelope as /analyze/profile with image_analysis = null."""
    try:
        logger.info("text_analysis_request", length=len(text))
        result = pipeline.analyze_text_only(text)
        return JSONResponse(content=_shape_response(result))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("text_analysis_failed", error=str(e), error_type=type(e).__name__)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Text analysis failed")


@app.exception_handler(RealityCheckException)
async def reality_check_exception_handler(request, exc: RealityCheckException):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message=str(exc),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.error("unhandled_exception", error=str(exc), type=type(exc).__name__)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ).dict(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=config.API.host,
        port=config.API.port,
        reload=config.API.reload,
    )
