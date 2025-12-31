"""
Base module interface for RealityCheck AI detection modules.

All detection modules (image, text, metadata) inherit from BaseModule
and implement standardized interfaces for preprocessing, prediction, and explanation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
from enum import Enum
import time


class PredictionClass(str, Enum):
    """Standard prediction classes across all modules"""
    REAL = "real"
    FAKE = "fake"
    AUTHENTIC = "authentic"
    SYNTHETIC = "synthetic"
    HUMAN = "human"
    AI = "ai"
    UNCERTAIN = "uncertain"


@dataclass
class ModuleResult:
    """
    Standardized result from any detection module.

    Attributes:
        score: Authenticity score in [0, 1] where 1 = authentic, 0 = fake
        confidence: Model confidence in [0, 1]
        prediction: Classification result ("real", "fake", "uncertain")
        explanation: Module-specific explanations (Grad-CAM, SHAP, etc.)
        metadata: Additional information (processing time, model version, etc.)
        raw_output: Raw model outputs (logits, probabilities)
    """
    score: float
    confidence: float
    prediction: str
    explanation: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_output: Optional[Any] = None

    def __post_init__(self):
        """Validate fields after initialization"""
        if not 0 <= self.score <= 1:
            raise ValueError(f"Score must be in [0, 1], got {self.score}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "score": float(self.score),
            "confidence": float(self.confidence),
            "prediction": self.prediction,
            "explanation": self.explanation,
            "metadata": self.metadata
        }


class BaseModule(ABC):
    """
    Abstract base class for all detection modules.

    Each module implements:
    1. Model loading (load_model)
    2. Input preprocessing (preprocess)
    3. Prediction (predict)
    4. Explanation generation (explain)

    Modules can be called directly via __call__ for end-to-end processing.
    """

    def __init__(self, config: Dict[str, Any], name: str = "BaseModule"):
        """
        Initialize module with configuration.

        Args:
            config: Configuration dictionary with model paths, hyperparameters, etc.
            name: Module name for logging and identification
        """
        self.config = config
        self.name = name
        self._model = None
        self._is_loaded = False
        self._device = config.get("device", "cpu")

    @abstractmethod
    def load_model(self) -> None:
        """
        Load model weights and initialize.

        This method should:
        1. Load pretrained weights from disk
        2. Initialize model architecture
        3. Move model to specified device (CPU/GPU)
        4. Set model to evaluation mode
        5. Set self._is_loaded = True

        Raises:
            FileNotFoundError: If model weights not found
            RuntimeError: If model loading fails
        """
        pass

    @abstractmethod
    def preprocess(self, input_data: Any) -> Any:
        """
        Preprocess input data for model inference.

        Args:
            input_data: Raw input (image path, text string, etc.)

        Returns:
            Preprocessed data ready for model (tensor, tokens, etc.)

        Raises:
            ValueError: If input data is invalid
        """
        pass

    @abstractmethod
    def predict(self, preprocessed_data: Any) -> ModuleResult:
        """
        Make prediction on preprocessed data.

        Args:
            preprocessed_data: Output from preprocess()

        Returns:
            ModuleResult with score, confidence, prediction, and metadata

        Raises:
            RuntimeError: If prediction fails
        """
        pass

    @abstractmethod
    def explain(self, input_data: Any, prediction: Any) -> Dict[str, Any]:
        """
        Generate explanations for prediction using XAI techniques.

        Args:
            input_data: Original input data
            prediction: Model prediction output

        Returns:
            Dictionary with explanation visualizations, feature importance, etc.
            Format depends on module:
            - Image: Grad-CAM heatmaps
            - Text: SHAP/LIME token importance
            - Metadata: Rule-based feature reports
        """
        pass

    def __call__(self, input_data: Any, explain: bool = True) -> ModuleResult:
        """
        End-to-end processing pipeline.

        This method orchestrates the complete analysis workflow:
        1. Lazy load model (only on first call)
        2. Preprocess input data
        3. Run prediction through model
        4. Generate explanations (Grad-CAM/SHAP/LIME)
        5. Add metadata and return results

        Args:
            input_data: Raw input data (image path, text string, etc.)
            explain: Whether to generate explanations (default: True)

        Returns:
            ModuleResult with predictions and explanations

        Example:
            >>> module = ImageAuthenticityModule(config)
            >>> result = module("path/to/image.jpg")
            >>> print(result.score, result.prediction)
        """
        start_time = time.time()

        # ========================================
        # STEP 1: Lazy Model Loading
        # ========================================
        # Only load model on first call to improve startup time
        # Subsequent calls reuse the loaded model
        if not self._is_loaded:
            self.load_model()

        # ========================================
        # STEP 2: Preprocessing
        # ========================================
        # Convert raw input to model-ready format
        # - Image: Load, resize, normalize, convert to tensor
        # - Text: Tokenize, encode, create attention mask
        try:
            preprocessed = self.preprocess(input_data)
        except Exception as e:
            raise ValueError(f"Preprocessing failed: {str(e)}")

        # ========================================
        # STEP 3: Prediction
        # ========================================
        # Run forward pass through model to get predictions
        # Returns ModuleResult with score, confidence, and prediction label
        try:
            result = self.predict(preprocessed)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

        # ========================================
        # STEP 4: Explanation Generation (Optional)
        # ========================================
        # Generate visual/textual explanations using XAI techniques:
        # - Image: Grad-CAM heatmaps showing important regions
        # - Text: SHAP/LIME token importance scores
        # - Metadata: Rule-based feature importance
        #
        # Only generate if:
        # 1. explain=True (user requested explanations)
        # 2. result.explanation is empty (not already generated in predict())
        if explain and not result.explanation:
            try:
                explanation = self.explain(input_data, result.raw_output)
                result.explanation = explanation
            except Exception as e:
                # Don't fail entire analysis if explanation fails
                # Just log error and continue
                result.metadata["explanation_error"] = str(e)

        # ========================================
        # STEP 5: Add Metadata
        # ========================================
        # Attach processing time and module name for monitoring
        result.metadata["processing_time_ms"] = (time.time() - start_time) * 1000
        result.metadata["module_name"] = self.name

        return result

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._is_loaded

    def get_device(self) -> str:
        """Get current device (cuda/cpu)"""
        return self._device

    def unload_model(self) -> None:
        """Unload model from memory (useful for memory management)"""
        self._model = None
        self._is_loaded = False

        # Force garbage collection for GPU memory
        import gc
        gc.collect()

        if self._device.startswith("cuda"):
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass


class ModelLoadError(Exception):
    """Raised when model loading fails"""
    pass


class PreprocessingError(Exception):
    """Raised when preprocessing fails"""
    pass


class PredictionError(Exception):
    """Raised when prediction fails"""
    pass


class ExplanationError(Exception):
    """Raised when explanation generation fails"""
    pass
