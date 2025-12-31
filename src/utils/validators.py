"""
Input validation utilities.
"""

import re
from pathlib import Path
from typing import Union
from PIL import Image

from src.utils.exceptions import ValidationError


def validate_image_file(image_path: Union[str, Path]) -> Path:
    """
    Validate image file exists and is a supported format.

    Args:
        image_path: Path to image file

    Returns:
        Validated Path object

    Raises:
        ValidationError: If file doesn't exist or unsupported format
    """
    path = Path(image_path)

    if not path.exists():
        raise ValidationError(f"Image file not found: {image_path}")

    if not path.is_file():
        raise ValidationError(f"Path is not a file: {image_path}")

    # Check file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    if path.suffix.lower() not in valid_extensions:
        raise ValidationError(
            f"Unsupported image format: {path.suffix}. "
            f"Supported formats: {valid_extensions}"
        )

    # Try to open image to verify it's valid
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as e:
        raise ValidationError(f"Invalid or corrupted image file: {e}")

    return path


def validate_text_input(text: str, min_length: int = 10, max_length: int = 1000) -> str:
    """
    Validate text input.

    Args:
        text: Input text to validate
        min_length: Minimum character length
        max_length: Maximum character length

    Returns:
        Validated text

    Raises:
        ValidationError: If text is invalid
    """
    if not isinstance(text, str):
        raise ValidationError(f"Text must be string, got {type(text)}")

    if not text or not text.strip():
        raise ValidationError("Text cannot be empty")

    text = text.strip()

    if len(text) < min_length:
        raise ValidationError(
            f"Text too short: {len(text)} chars (minimum: {min_length})"
        )

    if len(text) > max_length:
        raise ValidationError(
            f"Text too long: {len(text)} chars (maximum: {max_length})"
        )

    return text


def validate_url(url: str) -> str:
    """
    Validate URL format.

    Args:
        url: URL string

    Returns:
        Validated URL

    Raises:
        ValidationError: If URL is invalid
    """
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )

    if not url_pattern.match(url):
        raise ValidationError(f"Invalid URL format: {url}")

    return url


def validate_score(score: float, name: str = "score") -> float:
    """
    Validate score is in valid range [0, 1].

    Args:
        score: Score value
        name: Name of the score for error message

    Returns:
        Validated score

    Raises:
        ValidationError: If score is out of range
    """
    if not isinstance(score, (int, float)):
        raise ValidationError(f"{name} must be numeric, got {type(score)}")

    if not 0 <= score <= 1:
        raise ValidationError(f"{name} must be in [0, 1], got {score}")

    return float(score)


def validate_file_size(file_path: Union[str, Path], max_size_mb: int = 10) -> int:
    """
    Validate file size is within limit.

    Args:
        file_path: Path to file
        max_size_mb: Maximum file size in MB

    Returns:
        File size in bytes

    Raises:
        ValidationError: If file exceeds size limit
    """
    path = Path(file_path)

    if not path.exists():
        raise ValidationError(f"File not found: {file_path}")

    file_size = path.stat().st_size
    max_size_bytes = max_size_mb * 1024 * 1024

    if file_size > max_size_bytes:
        raise ValidationError(
            f"File too large: {file_size / 1024 / 1024:.2f}MB "
            f"(maximum: {max_size_mb}MB)"
        )

    return file_size
