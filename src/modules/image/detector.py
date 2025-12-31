"""
YOLOv8 face detection for image preprocessing.
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2
from pathlib import Path

from src.utils.logging import get_logger
from src.utils.exceptions import NoFaceDetectedError, ImageProcessingError

logger = get_logger(__name__)


class FaceBoundingBox:
    """Face bounding box with confidence"""

    def __init__(self, x1: int, y1: int, x2: int, y2: int, confidence: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)

    def to_dict(self) -> dict:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "confidence": float(self.confidence)
        }


class YOLOv8FaceDetector:
    """
    YOLOv8-based face detection.

    Uses ultralytics YOLOv8 for real-time face detection with high accuracy.
    """

    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """
        Initialize YOLOv8 face detector.

        Args:
            model_path: Path to YOLOv8 weights (uses default if None)
            confidence_threshold: Minimum confidence for face detection
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_path = model_path

    def load_model(self) -> None:
        """Load YOLOv8 model"""
        try:
            from ultralytics import YOLO

            if self.model_path and Path(self.model_path).exists():
                logger.info("loading_yolo_model", path=self.model_path)
                self.model = YOLO(self.model_path)
            else:
                # Use pretrained face detection model
                logger.info("loading_pretrained_yolo")
                # For now, use general YOLOv8n model
                # In production, use a face-specific model like yolov8n-face.pt
                self.model = YOLO('yolov8n.pt')

        except ImportError:
            raise ImportError(
                "ultralytics package required for YOLOv8. "
                "Install with: pip install ultralytics"
            )
        except Exception as e:
            raise ImageProcessingError(f"Failed to load YOLOv8 model: {e}")

    def detect_faces(self, image: np.ndarray) -> List[FaceBoundingBox]:
        """
        Detect faces in image.

        Args:
            image: Input image as numpy array (BGR format from cv2)

        Returns:
            List of FaceBoundingBox objects sorted by confidence

        Raises:
            NoFaceDetectedError: If no faces detected
            ImageProcessingError: If detection fails
        """
        if self.model is None:
            self.load_model()

        try:
            # Run inference
            results = self.model(image, verbose=False)

            faces = []
            for result in results:
                boxes = result.boxes

                for box in boxes:
                    # Get box coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])

                    # Filter by confidence and class (0 = person for COCO)
                    # For face-specific model, adjust class filtering
                    if confidence >= self.confidence_threshold:
                        face_box = FaceBoundingBox(
                            int(x1), int(y1), int(x2), int(y2), confidence
                        )
                        faces.append(face_box)

            if not faces:
                raise NoFaceDetectedError("No faces detected in image")

            # Sort by confidence (highest first)
            faces.sort(key=lambda x: x.confidence, reverse=True)

            logger.info("faces_detected", count=len(faces))
            return faces

        except NoFaceDetectedError:
            raise
        except Exception as e:
            raise ImageProcessingError(f"Face detection failed: {e}")

    def detect_largest_face(self, image: np.ndarray) -> FaceBoundingBox:
        """
        Detect and return the largest face (by area).

        Args:
            image: Input image

        Returns:
            Largest face bounding box

        Raises:
            NoFaceDetectedError: If no faces detected
        """
        faces = self.detect_faces(image)

        # Return largest face by area
        largest_face = max(faces, key=lambda x: x.area)
        return largest_face

    def crop_face(
        self,
        image: np.ndarray,
        bbox: FaceBoundingBox,
        padding: float = 0.2
    ) -> np.ndarray:
        """
        Crop face from image with optional padding.

        Args:
            image: Input image
            bbox: Face bounding box
            padding: Padding ratio to add around face (0.2 = 20% padding)

        Returns:
            Cropped face image
        """
        h, w = image.shape[:2]

        # Add padding
        pad_w = int(bbox.width * padding)
        pad_h = int(bbox.height * padding)

        x1 = max(0, bbox.x1 - pad_w)
        y1 = max(0, bbox.y1 - pad_h)
        x2 = min(w, bbox.x2 + pad_w)
        y2 = min(h, bbox.y2 + pad_h)

        cropped = image[y1:y2, x1:x2]
        return cropped


def detect_and_crop_face(
    image_path: str,
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.5,
    return_largest: bool = True
) -> Tuple[np.ndarray, FaceBoundingBox]:
    """
    Convenience function to detect and crop face from image file.

    Args:
        image_path: Path to image file
        model_path: Path to YOLOv8 model
        confidence_threshold: Minimum detection confidence
        return_largest: Return largest face if multiple detected

    Returns:
        Tuple of (cropped_face, bounding_box)

    Example:
        >>> face, bbox = detect_and_crop_face("profile.jpg")
        >>> print(f"Face confidence: {bbox.confidence:.2f}")
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ImageProcessingError(f"Failed to load image: {image_path}")

    # Detect faces
    detector = YOLOv8FaceDetector(model_path, confidence_threshold)

    if return_largest:
        bbox = detector.detect_largest_face(image)
    else:
        faces = detector.detect_faces(image)
        bbox = faces[0]  # Highest confidence

    # Crop face
    cropped = detector.crop_face(image, bbox)

    return cropped, bbox
