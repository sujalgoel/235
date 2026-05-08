"""Custom exceptions for RealityCheck AI."""


class RealityCheckException(Exception):
    """Base exception for the application."""


class ModelLoadError(RealityCheckException):
    """Raised when a model fails to load."""


class ImageProcessingError(RealityCheckException):
    """Raised when image preprocessing fails."""


class TextProcessingError(RealityCheckException):
    """Raised when text preprocessing fails."""


class PredictionError(RealityCheckException):
    """Raised when a model prediction fails."""
