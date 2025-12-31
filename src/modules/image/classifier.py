"""
ResNet-18 image authenticity classifier with Grad-CAM explainability.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
from pathlib import Path
import base64
from io import BytesIO

from src.modules.base import BaseModule, ModuleResult
from src.modules.image.detector import YOLOv8FaceDetector, NoFaceDetectedError
from src.modules.image.explainer import GradCAMExplainer
from src.utils.logging import get_logger
from src.utils.exceptions import ModelLoadError, ImageProcessingError, PredictionError

logger = get_logger(__name__)


class ResNet18FakeDetector(nn.Module):
    """
    ResNet-18 model for fake face detection.

    Architecture:
    - Pretrained ResNet-18 on ImageNet
    - Replace final FC layer for binary classification
    - Fine-tuned on real/fake face dataset
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load pretrained ResNet-18
        self.resnet = models.resnet18(pretrained=pretrained)

        # Replace final layer for binary classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)  # real, fake

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


class ImageAuthenticityModule(BaseModule):
    """
    Image authenticity detection module.

    Pipeline:
    1. YOLOv8 face detection
    2. Face cropping and preprocessing
    3. ResNet-18 classification
    4. Grad-CAM explanation generation
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, name="ImageAuthenticityModule")

        # Image parameters
        self.input_size = config.get("image_input_size", (128, 128))
        self.mean = config.get("image_mean", (0.485, 0.456, 0.406))
        self.std = config.get("image_std", (0.229, 0.224, 0.225))

        # Components
        self.face_detector = None
        self.classifier = None
        self.explainer = None

        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def load_model(self) -> None:
        """Load YOLOv8, ResNet-18, and Grad-CAM explainer"""
        try:
            logger.info("loading_image_models")

            # Load face detector
            yolo_path = self.config.get("yolo_model_path")
            self.face_detector = YOLOv8FaceDetector(model_path=yolo_path)
            self.face_detector.load_model()

            # Load classifier
            self.classifier = ResNet18FakeDetector(pretrained=True)

            model_path = self.config.get("image_model_path")
            if model_path and Path(model_path).exists():
                logger.info("loading_pretrained_weights", path=model_path)
                checkpoint = torch.load(model_path, map_location=self._device)

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.classifier.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.classifier.load_state_dict(checkpoint)

            # Move to device and set eval mode
            self.classifier.to(self._device)
            self.classifier.eval()

            # Initialize Grad-CAM explainer
            self.explainer = GradCAMExplainer(
                model=self.classifier.resnet,
                target_layer=self.classifier.resnet.layer4[-1]
            )

            self._is_loaded = True
            logger.info("image_models_loaded", device=self._device)

        except Exception as e:
            raise ModelLoadError(f"Failed to load image models: {e}")

    def preprocess(self, input_data: Any) -> Tuple[torch.Tensor, np.ndarray, Dict]:
        """
        Preprocess image for classification.

        Args:
            input_data: Image path (str) or numpy array

        Returns:
            Tuple of (preprocessed_tensor, original_face_crop, metadata)
        """
        try:
            # Load image
            if isinstance(input_data, (str, Path)):
                image = cv2.imread(str(input_data))
                if image is None:
                    raise ImageProcessingError(f"Failed to load image: {input_data}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(input_data, np.ndarray):
                image = input_data
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")

            # Detect face
            try:
                bbox = self.face_detector.detect_largest_face(image)
                face_crop = self.face_detector.crop_face(image, bbox)
            except NoFaceDetectedError:
                logger.warning("no_face_detected_using_full_image")
                face_crop = image
                bbox = None

            # Convert to PIL Image for transforms
            pil_image = Image.fromarray(face_crop)

            # Apply preprocessing
            tensor = self.transform(pil_image)
            tensor = tensor.unsqueeze(0)  # Add batch dimension

            metadata = {
                "face_detected": bbox is not None,
                "face_confidence": bbox.confidence if bbox else 0.0,
                "face_bbox": bbox.to_dict() if bbox else None,
                "original_size": image.shape[:2],
                "face_size": face_crop.shape[:2]
            }

            return tensor, face_crop, metadata

        except Exception as e:
            raise ImageProcessingError(f"Image preprocessing failed: {e}")

    def predict(self, preprocessed_data: Any) -> ModuleResult:
        """
        Classify face as real or fake.

        Args:
            preprocessed_data: Tuple from preprocess()

        Returns:
            ModuleResult with authenticity score and prediction
        """
        try:
            tensor, face_crop, metadata = preprocessed_data

            # Move to device
            tensor = tensor.to(self._device)

            # Forward pass
            with torch.no_grad():
                logits = self.classifier(tensor)
                probs = F.softmax(logits, dim=1)

            # Extract probabilities
            prob_real = float(probs[0, 0])
            prob_fake = float(probs[0, 1])

            # Authenticity score (probability of being real)
            score = prob_real
            confidence = max(prob_real, prob_fake)  # Confidence in prediction
            prediction = "real" if prob_real > prob_fake else "fake"

            # Add detection info to metadata
            metadata.update({
                "prob_real": prob_real,
                "prob_fake": prob_fake,
                "model_type": "ResNet-18"
            })

            return ModuleResult(
                score=score,
                confidence=confidence,
                prediction=prediction,
                explanation={},  # Will be filled by explain()
                metadata=metadata,
                raw_output={"logits": logits, "probs": probs, "face_crop": face_crop}
            )

        except Exception as e:
            raise PredictionError(f"Image prediction failed: {e}")

    def _numpy_to_base64_image(self, image: np.ndarray, format: str = 'PNG') -> str:
        """
        Convert numpy array to base64-encoded image string.

        Args:
            image: Numpy array (H, W, C) with values in [0, 255] or [0, 1]
            format: Image format (PNG, JPEG)

        Returns:
            Base64-encoded data URI string
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Convert to PIL Image
        pil_image = Image.fromarray(image)

        # Encode to base64
        buffer = BytesIO()
        pil_image.save(buffer, format=format)
        buffer.seek(0)
        base64_str = base64.b64encode(buffer.read()).decode('utf-8')

        return f"data:image/{format.lower()};base64,{base64_str}"

    def explain(self, input_data: Any, prediction: Any) -> Dict[str, Any]:
        """
        Generate Grad-CAM explanation.

        Args:
            input_data: Original input data
            prediction: Raw model output with face_crop

        Returns:
            Dictionary with Grad-CAM visualizations as base64 images
        """
        try:
            # Get face crop and prediction info
            face_crop = prediction.get("face_crop")
            probs = prediction.get("probs")

            if face_crop is None or probs is None:
                return {"error": "Missing data for explanation"}

            # Preprocess for Grad-CAM
            pil_image = Image.fromarray(face_crop)
            tensor = self.transform(pil_image).unsqueeze(0).to(self._device)

            # Target class for Grad-CAM (fake class = 1)
            target_class = torch.argmax(probs[0]).item()

            # Generate Grad-CAM
            heatmap = self.explainer.generate_heatmap(
                input_tensor=tensor,
                target_class=target_class
            )

            # Overlay on original image
            overlay = self.explainer.overlay_heatmap(face_crop, heatmap)

            # Identify key regions
            artifacts = self._identify_artifacts(heatmap)

            # Convert heatmap to colored visualization for display
            heatmap_uint8 = np.uint8(255 * heatmap)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            return {
                "grad_cam_heatmap": self._numpy_to_base64_image(heatmap_colored),
                "grad_cam_overlay": self._numpy_to_base64_image(overlay),
                "artifacts_detected": artifacts,
                "target_class": "fake" if target_class == 1 else "real"
            }

        except Exception as e:
            logger.error("explanation_failed", error=str(e))
            return {"error": str(e)}

    def _identify_artifacts(self, heatmap: np.ndarray, threshold: float = 0.7) -> list:
        """
        Identify suspicious regions from Grad-CAM heatmap.

        Args:
            heatmap: Grad-CAM heatmap [0, 1]
            threshold: Activation threshold

        Returns:
            List of identified artifacts
        """
        artifacts = []

        # Find high-activation regions
        high_activation = heatmap > threshold
        activation_ratio = np.mean(high_activation)

        if activation_ratio > 0.3:
            artifacts.append("Widespread suspicious patterns detected")

        # Check specific regions (eyes, mouth, etc.)
        h, w = heatmap.shape
        regions = {
            "upper": heatmap[:h//3, :],  # Forehead/eyes area
            "middle": heatmap[h//3:2*h//3, :],  # Nose/cheeks
            "lower": heatmap[2*h//3:, :]  # Mouth/chin
        }

        for region_name, region in regions.items():
            if np.max(region) > 0.8:
                artifacts.append(f"High activation in {region_name} region")

        if not artifacts:
            artifacts.append("No obvious artifacts detected")

        return artifacts
