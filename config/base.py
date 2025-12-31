"""
Base configuration for RealityCheck AI system.

This module provides centralized configuration management for all components:
- Model paths and hyperparameters
- API settings
- Fusion weights
- Device configuration
- Logging and monitoring
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import torch


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"


@dataclass
class ModelConfig:
    """Configuration for model paths and inference settings"""

    # Image Module
    image_model_path: str = str(MODELS_DIR / "image" / "resnet18_v1.0.pth")
    yolo_model_path: str = str(MODELS_DIR / "image" / "yolov8_face.pt")
    image_input_size: tuple = (128, 128)
    image_mean: tuple = (0.485, 0.456, 0.406)  # ImageNet stats
    image_std: tuple = (0.229, 0.224, 0.225)

    # Text Module
    text_model_path: str = str(MODELS_DIR / "text" / "distilbert_v1.0")
    text_model_name: str = "distilbert-base-uncased"  # For tokenizer
    max_sequence_length: int = 128
    text_batch_size: int = 16

    # Device Configuration
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    gpu_id: int = 0

    # Model Versioning
    model_version: str = "1.0.0"

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available, falling back to CPU")
            self.device = "cpu"


@dataclass
class FusionConfig:
    """
    Configuration for multimodal fusion.

    Weights for combining image, text, and metadata scores:
    Trust Score = w_I * I + w_T * T + w_M * M

    From paper: w_I = 0.4, w_T = 0.3, w_M = 0.3
    """
    image_weight: float = 0.4
    text_weight: float = 0.3
    metadata_weight: float = 0.3

    # Trust score interpretation thresholds
    very_low_threshold: float = 0.3  # < 0.3: Very low trust
    low_threshold: float = 0.5       # 0.3-0.5: Low trust
    moderate_threshold: float = 0.7  # 0.5-0.7: Moderate trust
    # >= 0.7: High trust

    # Ensemble strategy
    ensemble_method: str = "weighted"  # weighted, voting, stacking

    def __post_init__(self):
        """Validate fusion weights sum to 1.0"""
        total = self.image_weight + self.text_weight + self.metadata_weight
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(
                f"Fusion weights must sum to 1.0, got {total}. "
                f"(image: {self.image_weight}, text: {self.text_weight}, "
                f"metadata: {self.metadata_weight})"
            )

    def get_interpretation(self, trust_score: float) -> str:
        """Generate human-readable interpretation of trust score"""
        if trust_score < self.very_low_threshold:
            return "Very low trust - likely fake profile"
        elif trust_score < self.low_threshold:
            return "Low trust - suspicious indicators detected"
        elif trust_score < self.moderate_threshold:
            return "Moderate trust - mixed signals, manual review recommended"
        else:
            return "High trust - likely authentic profile"


@dataclass
class APIConfig:
    """Configuration for FastAPI web service"""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False  # Auto-reload on code changes (dev only)

    # Request limits
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: int = 300  # 5 minutes
    max_requests_per_minute: int = 100

    # CORS
    cors_origins: list = field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = True
    cors_allow_methods: list = field(default_factory=lambda: ["*"])
    cors_allow_headers: list = field(default_factory=lambda: ["*"])

    # Security
    secret_key: str = "change-this-in-production-use-env-var"
    api_key_enabled: bool = False
    rate_limit_enabled: bool = True

    # File storage
    upload_dir: str = str(PROJECT_ROOT / "uploads")
    visualization_dir: str = str(PROJECT_ROOT / "visualizations")


@dataclass
class MetadataConfig:
    """Configuration for metadata forensics module"""

    # ExifTool settings
    exiftool_path: Optional[str] = None  # Auto-detect if None

    # Expected EXIF fields for authentic images
    expected_fields: list = field(default_factory=lambda: [
        "Make", "Model", "DateTime", "DateTimeOriginal",
        "Software", "Orientation", "XResolution", "YResolution"
    ])

    # Scoring weights (from paper: α=0.5, β=0.3, γ=0.2)
    completeness_weight: float = 0.5  # α
    validity_weight: float = 0.3      # β
    anomaly_weight: float = 0.2       # γ

    # Suspicious software signatures
    suspicious_software: list = field(default_factory=lambda: [
        "Unknown", "GIMP", "Paint.NET", "Adobe Photoshop"  # May indicate editing
    ])


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring"""

    # Logging levels
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_format: str = "json"  # json, text
    log_file: str = str(LOGS_DIR / "realitycheck.log")

    # Monitoring
    prometheus_enabled: bool = False
    prometheus_port: int = 9090

    # Performance tracking
    track_inference_time: bool = True
    track_memory_usage: bool = True


@dataclass
class DataConfig:
    """Configuration for data paths and preprocessing"""

    # Data directories
    raw_dir: str = str(DATA_DIR / "raw")
    processed_dir: str = str(DATA_DIR / "processed")
    external_dir: str = str(DATA_DIR / "external")

    # Dataset splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Preprocessing
    image_augmentation: bool = True
    text_cleaning: bool = True


class Config:
    """
    Main configuration class that aggregates all sub-configs.

    Usage:
        from config.base import Config

        config = Config()
        print(config.MODEL.device)
        print(config.FUSION.image_weight)
    """

    def __init__(self, env: str = "development"):
        """
        Initialize configuration.

        Args:
            env: Environment name (development, production, testing)
        """
        self.ENV = env

        # Initialize all sub-configurations
        self.MODEL = ModelConfig()
        self.FUSION = FusionConfig()
        self.API = APIConfig()
        self.METADATA = MetadataConfig()
        self.LOGGING = LoggingConfig()
        self.DATA = DataConfig()

        # Create necessary directories
        self._create_directories()

        # Load environment variables if .env exists
        self._load_env_variables()

    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            LOGS_DIR,
            self.API.upload_dir,
            self.API.visualization_dir,
            MODELS_DIR / "image",
            MODELS_DIR / "text",
            MODELS_DIR / "fusion",
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _load_env_variables(self):
        """
        Load configuration from environment variables.

        Reads .env file in project root and overrides default config values.
        This allows deployment-specific configuration without code changes.

        Environment variables override defaults:
        - DEVICE: cpu/cuda
        - IMAGE_MODEL_PATH, TEXT_MODEL_PATH: Model file paths
        - IMAGE_WEIGHT, TEXT_WEIGHT, METADATA_WEIGHT: Fusion weights
        - API_HOST, API_PORT: Server settings
        - SECRET_KEY: API security key
        - LOG_LEVEL: Logging verbosity
        """
        try:
            from dotenv import load_dotenv
            load_dotenv()

            # ========================================
            # Model Configuration Overrides
            # ========================================
            self.MODEL.device = os.getenv("DEVICE", self.MODEL.device)
            self.MODEL.image_model_path = os.getenv("IMAGE_MODEL_PATH", self.MODEL.image_model_path)
            self.MODEL.text_model_path = os.getenv("TEXT_MODEL_PATH", self.MODEL.text_model_path)

            # ========================================
            # Fusion Weights Overrides
            # ========================================
            # Parse as floats since env vars are strings
            self.FUSION.image_weight = float(os.getenv("IMAGE_WEIGHT", self.FUSION.image_weight))
            self.FUSION.text_weight = float(os.getenv("TEXT_WEIGHT", self.FUSION.text_weight))
            self.FUSION.metadata_weight = float(os.getenv("METADATA_WEIGHT", self.FUSION.metadata_weight))

            # ========================================
            # API Configuration Overrides
            # ========================================
            self.API.host = os.getenv("API_HOST", self.API.host)
            self.API.port = int(os.getenv("API_PORT", self.API.port))  # Parse as int
            self.API.secret_key = os.getenv("SECRET_KEY", self.API.secret_key)

            # ========================================
            # Logging Configuration Overrides
            # ========================================
            self.LOGGING.log_level = os.getenv("LOG_LEVEL", self.LOGGING.log_level)

        except ImportError:
            # python-dotenv not installed, skip environment variable loading
            # This is acceptable in minimal deployments or testing
            pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "environment": self.ENV,
            "model": self.MODEL.__dict__,
            "fusion": self.FUSION.__dict__,
            "api": self.API.__dict__,
            "metadata": self.METADATA.__dict__,
            "logging": self.LOGGING.__dict__,
            "data": self.DATA.__dict__,
        }

    def __repr__(self) -> str:
        """String representation of configuration"""
        return f"Config(env={self.ENV}, device={self.MODEL.device})"


# Global configuration instance
config = Config()


# ========================================
# Environment-Specific Configuration Factory
# ========================================
def get_config(env: str = "development") -> Config:
    """
    Get configuration for specific environment.

    Creates a Config instance and applies environment-specific overrides
    for development, production, or testing environments.

    Args:
        env: Environment name (development, production, testing)

    Returns:
        Config instance with environment-specific settings

    Example:
        >>> config = get_config("production")
        >>> print(config.API.workers)  # 8 for production
        >>> print(config.LOGGING.log_level)  # WARNING for production
    """
    cfg = Config(env=env)

    if env == "production":
        # ========================================
        # PRODUCTION ENVIRONMENT
        # ========================================
        # Optimized for performance and security
        cfg.API.reload = False  # No auto-reload in production
        cfg.API.workers = 8  # Multi-worker for high throughput
        cfg.LOGGING.log_level = "WARNING"  # Less verbose logging
        cfg.API.api_key_enabled = True  # Require API keys for security
        cfg.API.rate_limit_enabled = True  # Prevent abuse

    elif env == "development":
        # ========================================
        # DEVELOPMENT ENVIRONMENT
        # ========================================
        # Optimized for rapid iteration and debugging
        cfg.API.reload = True  # Auto-reload on code changes
        cfg.API.workers = 1  # Single worker for easier debugging
        cfg.LOGGING.log_level = "DEBUG"  # Verbose logging for development
        cfg.API.cors_origins = ["http://localhost:3000"]  # Allow frontend dev server

    elif env == "testing":
        # ========================================
        # TESTING ENVIRONMENT
        # ========================================
        # Optimized for automated testing
        cfg.LOGGING.log_level = "ERROR"  # Minimal logging in tests
        cfg.API.port = 8001  # Different port to avoid conflicts
        cfg.MODEL.device = "cpu"  # Always use CPU for reproducibility

    return cfg
