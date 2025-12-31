"""
Custom exceptions for RealityCheck AI system.
"""


class RealityCheckException(Exception):
    """Base exception for all custom errors"""
    pass


class ModelLoadError(RealityCheckException):
    """Failed to load model weights"""
    pass


class ImageProcessingError(RealityCheckException):
    """Error during image processing"""
    pass


class NoFaceDetectedError(ImageProcessingError):
    """No face detected in image"""
    pass


class TextProcessingError(RealityCheckException):
    """Error during text processing"""
    pass


class MetadataExtractionError(RealityCheckException):
    """Error extracting metadata from image"""
    pass


class PredictionError(RealityCheckException):
    """Error during model prediction"""
    pass


class ExplanationError(RealityCheckException):
    """Error generating explanation"""
    pass


class ConfigurationError(RealityCheckException):
    """Invalid configuration"""
    pass


class ValidationError(RealityCheckException):
    """Input validation failed"""
    pass
