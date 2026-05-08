"""Production environment configuration"""

from config.base import Config
import os


class ProductionConfig(Config):
    """Configuration for production environment"""

    def __init__(self):
        super().__init__(env="production")

        # API settings for production
        self.API.reload = False
        self.API.workers = 8  # Multiple workers for better throughput
        cors_env = os.getenv("CORS_ORIGINS")
        if not cors_env:
            raise ValueError("CORS_ORIGINS must be set in production (comma-separated allowed origins)")
        self.API.cors_origins = [o.strip() for o in cors_env.split(",") if o.strip()]

        self.API.api_key_enabled = True
        self.API.rate_limit_enabled = True
        self.API.max_requests_per_minute = 100

        # Logging
        self.LOGGING.log_level = "WARNING"
        self.LOGGING.log_format = "json"  # Structured logging for monitoring
        self.LOGGING.prometheus_enabled = True

        # Model optimization
        # Enable model quantization or TorchScript compilation if needed
        # self.MODEL.use_quantization = True

        # Performance
        self.LOGGING.track_inference_time = True
        self.LOGGING.track_memory_usage = True


config = ProductionConfig()
