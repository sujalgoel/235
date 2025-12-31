"""Development environment configuration"""

from config.base import Config


class DevelopmentConfig(Config):
    """Configuration for development environment"""

    def __init__(self):
        super().__init__(env="development")

        # API settings for development
        self.API.reload = True  # Auto-reload on code changes
        self.API.workers = 1    # Single worker for debugging
        self.API.cors_origins = ["http://localhost:3000", "http://localhost:3001"]

        # Logging
        self.LOGGING.log_level = "DEBUG"
        self.LOGGING.log_format = "text"  # Easier to read during development

        # Use smaller models for faster iteration (optional)
        # self.MODEL.text_model_name = "distilbert-base-uncased"  # Already default

        # Disable rate limiting in development
        self.API.rate_limit_enabled = False
        self.API.api_key_enabled = False


config = DevelopmentConfig()
