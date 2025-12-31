"""
Hive AI cloud-based deepfake detection with explainability.

Uses Hive AI's Deepfake Detection API for state-of-the-art accuracy.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import requests
import os
from PIL import Image
from pathlib import Path
import base64
from io import BytesIO
from requests.auth import HTTPBasicAuth

from src.modules.base import BaseModule, ModuleResult
from src.utils.logging import get_logger
from src.utils.exceptions import ModelLoadError, ImageProcessingError, PredictionError

logger = get_logger(__name__)


class HiveAIImageDetector(BaseModule):
    """
    Cloud-based image deepfake detection using Hive AI.

    Features:
    - 95%+ accuracy on AI-generated images
    - Detects: DALL-E, Midjourney, Stable Diffusion, deepfakes
    - Real-time inference
    - No local GPU required
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, name="HiveAIImageDetector")

        # API configuration
        self.api_key = config.get("hive_api_key") or os.getenv("HIVE_API_KEY")
        self.api_url = "https://api.thehive.ai/api/v2/task/sync"

        if not self.api_key:
            raise ValueError(
                "Hive AI API key not found. Set HIVE_API_KEY in environment or config.\n"
                "Get your API key at: https://thehive.ai/pricing"
            )

        # Hive API uses Token authentication
        # If API key contains colon (username:password format), try using the full key first
        # Some API providers use only the first part as the token
        self.auth_type = 'token'
        logger.info("hive_api_key_format", has_colon=':' in self.api_key, length=len(self.api_key))

    def load_model(self) -> None:
        """Validate API connection"""
        try:
            logger.info("validating_hive_api_connection", auth_type=self.auth_type)

            # No actual test call - we'll validate on first real request
            # to avoid unnecessary API usage

            self._is_loaded = True
            logger.info("hive_api_ready", auth_type=self.auth_type)

        except Exception as e:
            raise ModelLoadError(f"Failed to validate Hive AI API: {e}")

    def preprocess(self, input_data: Any) -> Tuple[str, Dict]:
        """
        Prepare image for Hive AI API.

        Args:
            input_data: Image path (str) or numpy array

        Returns:
            Tuple of (image_path, metadata)
        """
        try:
            # Handle different input types
            if isinstance(input_data, (str, Path)):
                image_path = str(input_data)
                # Load to get metadata
                image = Image.open(image_path)
            elif isinstance(input_data, np.ndarray):
                # Save numpy array to temporary file
                import tempfile
                image = Image.fromarray(input_data)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                image.save(temp_file.name)
                image_path = temp_file.name
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")

            metadata = {
                "image_size": image.size,
                "image_format": image.format,
                "image_mode": image.mode
            }

            return image_path, metadata

        except Exception as e:
            raise ImageProcessingError(f"Image preprocessing failed: {e}")

    def predict(self, preprocessed_data: Any) -> ModuleResult:
        """
        Detect deepfakes using Hive AI API.

        Args:
            preprocessed_data: Tuple from preprocess()

        Returns:
            ModuleResult with authenticity score and prediction
        """
        try:
            image_path, metadata = preprocessed_data

            # Prepare request headers
            headers = {
                "Authorization": f"Token {self.api_key}"
            }

            # Open image file
            with open(image_path, 'rb') as image_file:
                files = {
                    'media': image_file
                }

                # Request deepfake detection
                data = {
                    'classes': 'ai_generated_image,deepfake'  # Detect both AI images and deepfakes
                }

                logger.info("calling_hive_api", path=image_path)

                response = requests.post(
                    self.api_url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=30
                )

            # Handle API response
            logger.info("hive_api_response", status_code=response.status_code)

            if response.status_code != 200:
                logger.error("hive_api_error",
                           status=response.status_code,
                           response_text=response.text,
                           headers=response.headers)
                raise PredictionError(
                    f"Hive API error {response.status_code}: {response.text}\n"
                    "Check your API key and quota at https://thehive.ai"
                )

            result = response.json()
            logger.info("hive_api_response_received", response_keys=list(result.keys()))

            # Parse Hive AI response
            score, confidence, prediction, explanation = self._parse_hive_response(result)

            # Update metadata
            metadata.update({
                "model_type": "Hive AI Deepfake Detector",
                "api_version": "v2",
                "classes_detected": explanation.get("detected_classes", [])
            })

            return ModuleResult(
                score=score,
                confidence=confidence,
                prediction=prediction,
                explanation=explanation,
                metadata=metadata,
                raw_output=result
            )

        except requests.exceptions.RequestException as e:
            raise PredictionError(f"Hive API request failed: {e}")
        except Exception as e:
            raise PredictionError(f"Image prediction failed: {e}")

    def _parse_hive_response(self, response: Dict[str, Any]) -> Tuple[float, float, str, Dict]:
        """
        Parse Hive AI API response.

        Hive returns confidence scores for different classes.
        We extract AI-generated and deepfake scores.

        Args:
            response: Hive API JSON response

        Returns:
            Tuple of (score, confidence, prediction, explanation)
        """
        try:
            logger.info("parsing_hive_response", response_structure=response.keys() if isinstance(response, dict) else type(response))

            status = response.get('status', [])
            logger.info("hive_status_check", has_status=bool(status), status_len=len(status) if status else 0)

            if not status or status[0].get('response', {}).get('status') != 'success':
                logger.error("hive_unsuccessful_status",
                           has_status=bool(status),
                           status_content=status[0] if status else None)
                raise ValueError("Hive API returned unsuccessful status")

            # Extract classification results
            output = status[0].get('response', {}).get('output', [])

            # Look for AI-generated and deepfake scores
            ai_score = 0.0
            deepfake_score = 0.0
            detected_classes = []

            for item in output:
                for class_result in item.get('classes', []):
                    class_name = class_result.get('class', '')
                    class_score = class_result.get('score', 0.0)

                    if 'ai_generated' in class_name.lower():
                        ai_score = max(ai_score, class_score)
                        if class_score > 0.5:
                            detected_classes.append(f"AI-generated ({class_score:.2%})")

                    if 'deepfake' in class_name.lower():
                        deepfake_score = max(deepfake_score, class_score)
                        if class_score > 0.5:
                            detected_classes.append(f"Deepfake ({class_score:.2%})")

            # Combined fake score (max of AI-generated and deepfake)
            fake_score = max(ai_score, deepfake_score)

            # Authenticity score (inverse of fake score)
            authenticity_score = 1.0 - fake_score

            # Confidence is the absolute distance from 0.5
            confidence = abs(authenticity_score - 0.5) * 2

            # Prediction
            prediction = "real" if authenticity_score > 0.5 else "fake"

            # Explanation
            explanation = {
                "ai_generated_score": ai_score,
                "deepfake_score": deepfake_score,
                "detected_classes": detected_classes,
                "detection_method": "Hive AI Multi-Model Ensemble"
            }

            if fake_score > 0.7:
                explanation["verdict"] = "High confidence AI-generated or deepfake"
            elif fake_score > 0.5:
                explanation["verdict"] = "Likely AI-generated or deepfake"
            elif fake_score > 0.3:
                explanation["verdict"] = "Suspicious - manual review recommended"
            else:
                explanation["verdict"] = "Likely authentic image"

            return authenticity_score, confidence, prediction, explanation

        except Exception as e:
            logger.error("failed_to_parse_hive_response",
                       error=str(e),
                       error_type=type(e).__name__,
                       response_sample=str(response)[:500])
            # Fallback: assume uncertain
            return 0.5, 0.0, "unknown", {"error": str(e), "response_preview": str(response)[:200]}

    def explain(self, input_data: Any, prediction: Any) -> Dict[str, Any]:
        """
        Generate explanation from Hive AI results.

        Args:
            input_data: Original input data
            prediction: Raw Hive API output

        Returns:
            Dictionary with explanation details
        """
        # Hive AI explanations are already included in the prediction
        # We just return the structured explanation

        try:
            status = prediction.get('status', [])
            if not status:
                return {"error": "No status in Hive response"}

            output = status[0].get('response', {}).get('output', [])

            explanation = {
                "provider": "Hive AI",
                "detection_classes": [],
                "confidence_scores": {}
            }

            for item in output:
                for class_result in item.get('classes', []):
                    class_name = class_result.get('class', '')
                    class_score = class_result.get('score', 0.0)

                    explanation["confidence_scores"][class_name] = class_score

                    if class_score > 0.3:
                        explanation["detection_classes"].append({
                            "class": class_name,
                            "score": class_score,
                            "threshold": 0.5
                        })

            return explanation

        except Exception as e:
            logger.error("explanation_generation_failed", error=str(e))
            return {"error": str(e)}
