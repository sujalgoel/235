"""Base interface for detection modules."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModuleResult:
    """Standardised return type from any detection module.

    Attributes:
        score: Authenticity score in [0, 1]; 1 = authentic, 0 = fake/AI.
        confidence: Model confidence in [0, 1].
        prediction: Class label (e.g. "real", "fake", "human", "ai").
        explanation: Module-specific explanations (Grad-CAM, LIME, …).
        metadata: Processing details (model name, timing, …).
        raw_output: Raw model outputs for downstream consumers.
    """

    score: float
    confidence: float
    prediction: str
    explanation: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_output: Optional[Any] = None

    def __post_init__(self):
        if not 0 <= self.score <= 1:
            raise ValueError(f"Score must be in [0, 1], got {self.score}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": float(self.score),
            "confidence": float(self.confidence),
            "prediction": self.prediction,
            "explanation": self.explanation,
            "metadata": self.metadata,
        }


class BaseModule(ABC):
    """Abstract base for image / text detection modules.

    Subclasses implement load_model / preprocess / predict / explain.
    `__call__` runs the lazy-load → preprocess → predict → explain flow.
    """

    def __init__(self, config: Dict[str, Any], name: str = "BaseModule"):
        self.config = config
        self.name = name
        self._is_loaded = False
        self._device = config.get("device", "cpu")

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def preprocess(self, input_data: Any) -> Any: ...

    @abstractmethod
    def predict(self, preprocessed_data: Any) -> ModuleResult: ...

    @abstractmethod
    def explain(self, input_data: Any, prediction: Any) -> Dict[str, Any]: ...

    def __call__(self, input_data: Any, explain: bool = True) -> ModuleResult:
        start = time.time()

        if not self._is_loaded:
            self.load_model()

        try:
            preprocessed = self.preprocess(input_data)
        except Exception as e:
            raise ValueError(f"Preprocessing failed: {e}")

        try:
            result = self.predict(preprocessed)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

        if explain and not result.explanation:
            try:
                result.explanation = self.explain(input_data, result.raw_output)
            except Exception as e:
                # Explanation is best-effort; don't fail the whole prediction.
                result.metadata["explanation_error"] = str(e)

        result.metadata["processing_time_ms"] = (time.time() - start) * 1000
        result.metadata["module_name"] = self.name
        return result

    def is_loaded(self) -> bool:
        return self._is_loaded
