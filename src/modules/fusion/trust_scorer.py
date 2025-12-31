"""
Trust Score computation for multimodal fusion.

Implements the weighted fusion strategy from the paper:
Trust Score = w_I * I + w_T * T + w_M * M

Where:
- w_I = 0.4 (image weight)
- w_T = 0.3 (text weight)
- w_M = 0.3 (metadata weight)
- I, T, M are authenticity scores from each module ∈ [0, 1]
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from enum import Enum

from src.modules.base import ModuleResult
from config.base import FusionConfig


class TrustLevel(str, Enum):
    """Trust level categories based on trust score"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass
class TrustScoreResult:
    """
    Result of trust score computation.

    Attributes:
        trust_score: Final weighted trust score ∈ [0, 1]
        trust_level: Categorical trust level
        interpretation: Human-readable interpretation
        module_scores: Individual scores from each module
        module_confidences: Confidence from each module
        contributing_factors: Key factors influencing the score
        metadata: Additional information
    """
    trust_score: float
    trust_level: TrustLevel
    interpretation: str
    module_scores: Dict[str, float]
    module_confidences: Dict[str, float]
    contributing_factors: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "trust_score": float(self.trust_score),
            "trust_level": self.trust_level.value,
            "interpretation": self.interpretation,
            "module_scores": self.module_scores,
            "module_confidences": self.module_confidences,
            "contributing_factors": self.contributing_factors,
            "metadata": self.metadata
        }


class TrustScorer:
    """
    Multimodal fusion with weighted combination.

    Combines predictions from image, text, and metadata modules
    into a unified trust score using configurable weights.
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        """
        Initialize trust scorer.

        Args:
            config: Fusion configuration (uses defaults if None)
        """
        self.config = config or FusionConfig()

        # Weights from paper
        self.weights = {
            'image': self.config.image_weight,
            'text': self.config.text_weight,
            'metadata': self.config.metadata_weight
        }

        # Trust level thresholds
        self.thresholds = {
            'very_low': self.config.very_low_threshold,
            'low': self.config.low_threshold,
            'moderate': self.config.moderate_threshold
        }

    def compute_trust_score(
        self,
        image_result: Optional[ModuleResult] = None,
        text_result: Optional[ModuleResult] = None,
        metadata_result: Optional[ModuleResult] = None
    ) -> TrustScoreResult:
        """
        Compute weighted trust score from module results.

        Implements Equation 1 from paper:
        Trust Score = w_I * I + w_T * T + w_M * M

        Args:
            image_result: Result from image authenticity module
            text_result: Result from text authenticity module
            metadata_result: Result from metadata forensics module

        Returns:
            TrustScoreResult with final score and interpretation

        Raises:
            ValueError: If all module results are None
        """
        # ========================================
        # STEP 1: Collect scores from available modalities
        # ========================================
        # Extract authenticity scores and confidence values from each module
        # Scores range from 0 (definitely fake) to 1 (definitely real)
        scores = {}
        confidences = {}

        # Image module: AI-generated image detection
        # Score represents probability of being a real/authentic image
        if image_result is not None:
            scores['image'] = image_result.score
            confidences['image'] = image_result.confidence

        # Text module: AI-written text detection
        # Score represents probability of being human-written text
        if text_result is not None:
            scores['text'] = text_result.score
            confidences['text'] = text_result.confidence

        # Metadata module: EXIF forensics (always None in current implementation)
        # Would analyze camera metadata to detect synthetic images
        if metadata_result is not None:
            scores['metadata'] = metadata_result.score
            confidences['metadata'] = metadata_result.confidence

        # Validation: At least one module must have succeeded
        # Can't compute trust score with zero inputs
        if not scores:
            raise ValueError("At least one module result must be provided")

        # ========================================
        # STEP 2: Adjust fusion weights for missing modules
        # ========================================
        # If metadata is missing (common), redistribute its weight to image/text
        # Example: If metadata (0.3) is missing:
        #   Original: image=0.4, text=0.3, metadata=0.3
        #   Adjusted: image=0.57 (0.4/0.7), text=0.43 (0.3/0.7)
        # This ensures weights always sum to 1.0
        adjusted_weights = self._adjust_weights(scores.keys())

        # ========================================
        # STEP 3: Compute weighted trust score
        # ========================================
        # Implements Equation 1 from research paper:
        # Trust Score = w_I × I + w_T × T + w_M × M
        # Where w_I, w_T, w_M are weights and I, T, M are module scores
        #
        # This is a weighted average that combines:
        # - Image authenticity (how real does the image look?)
        # - Text authenticity (how human does the text sound?)
        # - Metadata forensics (do EXIF properties look legitimate?)
        trust_score = sum(
            adjusted_weights[module] * score
            for module, score in scores.items()
        )

        # ========================================
        # STEP 4: Categorize trust level
        # ========================================
        # Convert numerical score to categorical label
        # Thresholds: <0.3=VERY_LOW, <0.5=LOW, <0.7=MODERATE, >=0.7=HIGH
        trust_level = self._determine_trust_level(trust_score)

        # ========================================
        # STEP 5: Generate human-readable interpretation
        # ========================================
        # Create explanation text based on:
        # - Overall trust score
        # - Agreement between modules (do they all agree or conflict?)
        # - Average confidence (are the models certain or uncertain?)
        interpretation = self._generate_interpretation(
            trust_score,
            trust_level,
            scores,
            confidences
        )

        # ========================================
        # STEP 6: Extract contributing factors
        # ========================================
        # Pull out specific reasons why the score is high/low:
        # - Image: "Synthetic image artifacts detected"
        # - Text: "AI-generated text patterns detected"
        # - Metadata: "Missing or suspicious EXIF data"
        contributing_factors = self._identify_contributing_factors(
            image_result,
            text_result,
            metadata_result
        )

        # ========================================
        # STEP 7: Compile metadata for debugging
        # ========================================
        # Store additional information useful for analysis:
        # - Which weights were actually used (after adjustment)
        # - Which modules were available
        # - Average confidence across all modules
        metadata = {
            "weights_used": adjusted_weights,
            "modules_available": list(scores.keys()),
            "average_confidence": np.mean(list(confidences.values()))
        }

        return TrustScoreResult(
            trust_score=trust_score,
            trust_level=trust_level,
            interpretation=interpretation,
            module_scores=scores,
            module_confidences=confidences,
            contributing_factors=contributing_factors,
            metadata=metadata
        )

    def _adjust_weights(self, available_modules: set) -> Dict[str, float]:
        """
        Adjust fusion weights when some modules are missing.

        If a module is unavailable, redistribute its weight proportionally
        to the other modules.

        Args:
            available_modules: Set of module names that provided results

        Returns:
            Adjusted weights that sum to 1.0
        """
        adjusted = {}
        total_available_weight = sum(
            self.weights[module]
            for module in available_modules
        )

        for module in available_modules:
            adjusted[module] = self.weights[module] / total_available_weight

        return adjusted

    def _determine_trust_level(self, trust_score: float) -> TrustLevel:
        """
        Determine categorical trust level from numerical score.

        Args:
            trust_score: Trust score ∈ [0, 1]

        Returns:
            TrustLevel enum
        """
        if trust_score < self.thresholds['very_low']:
            return TrustLevel.VERY_LOW
        elif trust_score < self.thresholds['low']:
            return TrustLevel.LOW
        elif trust_score < self.thresholds['moderate']:
            return TrustLevel.MODERATE
        else:
            return TrustLevel.HIGH

    def _generate_interpretation(
        self,
        trust_score: float,
        trust_level: TrustLevel,
        scores: Dict[str, float],
        confidences: Dict[str, float]
    ) -> str:
        """
        Generate human-readable interpretation of trust score.

        Args:
            trust_score: Final trust score
            trust_level: Categorical trust level
            scores: Individual module scores
            confidences: Individual module confidences

        Returns:
            Interpretation string
        """
        # Base interpretation from trust level
        base_interpretation = self.config.get_interpretation(trust_score)

        # Add details about module agreement
        score_values = list(scores.values())
        if len(score_values) > 1:
            score_std = np.std(score_values)

            if score_std < 0.1:
                agreement = "All modules strongly agree."
            elif score_std < 0.2:
                agreement = "Modules generally agree."
            else:
                agreement = "Modules show conflicting signals - manual review recommended."

            base_interpretation += f" {agreement}"

        # Add confidence note
        avg_confidence = np.mean(list(confidences.values()))
        if avg_confidence < 0.6:
            base_interpretation += " Note: Low confidence across modules."

        return base_interpretation

    def _identify_contributing_factors(
        self,
        image_result: Optional[ModuleResult],
        text_result: Optional[ModuleResult],
        metadata_result: Optional[ModuleResult]
    ) -> Dict[str, List[str]]:
        """
        Identify key factors contributing to trust score.

        Extracts important findings from each module's explanation.

        Args:
            image_result: Image module result
            text_result: Text module result
            metadata_result: Metadata module result

        Returns:
            Dictionary of contributing factors by module
        """
        factors = {}

        # Image factors
        if image_result is not None:
            factors['image'] = []
            if image_result.score < 0.5:
                factors['image'].append("Synthetic image artifacts detected")
            if image_result.explanation:
                # Extract key findings from Grad-CAM
                artifacts = image_result.explanation.get("artifacts_detected", [])
                factors['image'].extend(artifacts)

        # Text factors
        if text_result is not None:
            factors['text'] = []
            if text_result.score < 0.5:
                factors['text'].append("AI-generated text patterns detected")
            if text_result.explanation:
                # Extract key linguistic patterns
                patterns = text_result.explanation.get("ai_indicators", [])
                factors['text'].extend(patterns[:3])  # Top 3

        # Metadata factors
        if metadata_result is not None:
            factors['metadata'] = []
            if metadata_result.score < 0.5:
                factors['metadata'].append("Missing or suspicious EXIF data")
            if metadata_result.explanation:
                # Extract anomalies
                anomalies = metadata_result.explanation.get("anomalies", [])
                factors['metadata'].extend([a.get("description", "") for a in anomalies])

        return factors

    def calibrate_score(self, raw_score: float, confidence: float) -> float:
        """
        Apply confidence-based calibration to score.

        Lower confidence scores are adjusted towards 0.5 (uncertain).

        Args:
            raw_score: Raw prediction score
            confidence: Model confidence

        Returns:
            Calibrated score
        """
        # Calibration: pull low-confidence scores towards 0.5
        calibrated = raw_score * confidence + 0.5 * (1 - confidence)
        return calibrated

    def ensemble_predictions(
        self,
        predictions: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Ensemble multiple predictions with uncertainty weighting.

        Args:
            predictions: List of (score, confidence) tuples

        Returns:
            Tuple of (ensemble_score, ensemble_confidence)
        """
        if not predictions:
            raise ValueError("No predictions to ensemble")

        # Weight by confidence
        scores = np.array([p[0] for p in predictions])
        confidences = np.array([p[1] for p in predictions])

        weighted_score = np.average(scores, weights=confidences)
        ensemble_confidence = np.mean(confidences)

        return float(weighted_score), float(ensemble_confidence)


def fuse_multimodal_results(
    image_result: Optional[ModuleResult] = None,
    text_result: Optional[ModuleResult] = None,
    metadata_result: Optional[ModuleResult] = None,
    config: Optional[FusionConfig] = None
) -> TrustScoreResult:
    """
    Convenience function for computing trust score.

    Args:
        image_result: Image module result
        text_result: Text module result
        metadata_result: Metadata module result
        config: Fusion configuration

    Returns:
        TrustScoreResult

    Example:
        >>> from src.modules.fusion.trust_scorer import fuse_multimodal_results
        >>> result = fuse_multimodal_results(
        ...     image_result=img_result,
        ...     text_result=txt_result,
        ...     metadata_result=meta_result
        ... )
        >>> print(f"Trust Score: {result.trust_score:.2f}")
        >>> print(f"Interpretation: {result.interpretation}")
    """
    scorer = TrustScorer(config=config)
    return scorer.compute_trust_score(
        image_result=image_result,
        text_result=text_result,
        metadata_result=metadata_result
    )
