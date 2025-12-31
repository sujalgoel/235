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
        self.API.cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")

        # Security - MUST be set via environment variables
        self.API.secret_key = os.getenv("SECRET_KEY")
        if not self.API.secret_key or self.API.secret_key == "change-this-in-production-use-env-var":
            raise ValueError("SECRET_KEY must be set in production environment")

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
