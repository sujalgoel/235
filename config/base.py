"""Centralised configuration for RealityCheck AI."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"


@dataclass
class ModelConfig:
    """Model paths and inference settings."""

    image_model_path: str = str(MODELS_DIR / "image" / "resnet18_v1.0.pth")
    text_model_path: str = str(MODELS_DIR / "text" / "distilbert_v1.0")

    image_input_size: tuple = (128, 128)
    image_mean: tuple = (0.485, 0.456, 0.406)
    image_std: tuple = (0.229, 0.224, 0.225)

    max_sequence_length: int = 128
    text_batch_size: int = 16

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    gpu_id: int = 0

    def __post_init__(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available, falling back to CPU")
            self.device = "cpu"


@dataclass
class FusionConfig:
    """Trust-score fusion weights and band thresholds.

    Trust Score = w_I * I + w_T * T + w_M * M  (paper defaults: 0.4 / 0.3 / 0.3)
    """

    image_weight: float = 0.4
    text_weight: float = 0.3
    metadata_weight: float = 0.3

    very_low_threshold: float = 0.3
    low_threshold: float = 0.5
    moderate_threshold: float = 0.7

    def __post_init__(self):
        total = self.image_weight + self.text_weight + self.metadata_weight
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Fusion weights must sum to 1.0, got {total} "
                f"(image={self.image_weight}, text={self.text_weight}, "
                f"metadata={self.metadata_weight})"
            )

    def get_interpretation(self, trust_score: float) -> str:
        if trust_score < self.very_low_threshold:
            return "Very low trust - likely fake profile"
        if trust_score < self.low_threshold:
            return "Low trust - suspicious indicators detected"
        if trust_score < self.moderate_threshold:
            return "Moderate trust - mixed signals, manual review recommended"
        return "High trust - likely authentic profile"


@dataclass
class APIConfig:
    """FastAPI server settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False

    max_upload_size: int = 10 * 1024 * 1024  # 10 MiB

    # CORS — safe defaults; per-env configs widen as needed.
    cors_origins: list = field(default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"])
    cors_allow_credentials: bool = False
    cors_allow_methods: list = field(default_factory=lambda: ["GET", "POST", "OPTIONS"])
    cors_allow_headers: list = field(default_factory=lambda: ["Content-Type", "Authorization"])

    upload_dir: str = str(PROJECT_ROOT / "uploads")
    visualization_dir: str = str(PROJECT_ROOT / "visualizations")


@dataclass
class LoggingConfig:
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: str = str(LOGS_DIR / "realitycheck.log")


class Config:
    """Aggregate of all sub-configs, with .env overrides applied at construction."""

    def __init__(self, env: str = "development"):
        self.ENV = env
        self.MODEL = ModelConfig()
        self.FUSION = FusionConfig()
        self.API = APIConfig()
        self.LOGGING = LoggingConfig()

        self._create_directories()
        self._load_env_variables()

    def _create_directories(self):
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
        """Apply DEVICE / API_HOST / API_PORT / *_MODEL_PATH / *_WEIGHT / LOG_LEVEL overrides."""
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass  # python-dotenv is optional

        self.MODEL.device = os.getenv("DEVICE", self.MODEL.device)
        self.MODEL.image_model_path = os.getenv("IMAGE_MODEL_PATH", self.MODEL.image_model_path)
        self.MODEL.text_model_path = os.getenv("TEXT_MODEL_PATH", self.MODEL.text_model_path)

        self.FUSION.image_weight = float(os.getenv("IMAGE_WEIGHT", self.FUSION.image_weight))
        self.FUSION.text_weight = float(os.getenv("TEXT_WEIGHT", self.FUSION.text_weight))
        self.FUSION.metadata_weight = float(os.getenv("METADATA_WEIGHT", self.FUSION.metadata_weight))

        self.API.host = os.getenv("API_HOST", self.API.host)
        port_value = int(os.getenv("API_PORT", self.API.port))
        if not 1 <= port_value <= 65535:
            raise ValueError(f"API_PORT must be in [1, 65535], got {port_value}")
        self.API.port = port_value

        self.LOGGING.log_level = os.getenv("LOG_LEVEL", self.LOGGING.log_level)


def get_config(env: str = "development") -> Config:
    """Build a Config instance with environment-specific overrides."""
    cfg = Config(env=env)

    if env == "production":
        cfg.API.reload = False
        cfg.API.workers = 8
        cfg.LOGGING.log_level = "WARNING"
    elif env == "development":
        cfg.API.reload = True
        cfg.API.workers = 1
        cfg.LOGGING.log_level = "DEBUG"
    elif env == "testing":
        cfg.LOGGING.log_level = "ERROR"
        cfg.API.port = 8001
        cfg.MODEL.device = "cpu"

    return cfg
