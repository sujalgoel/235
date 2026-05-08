"""Trust-score fusion edge cases."""

import math

import pytest

from src.modules.base import ModuleResult
from src.modules.fusion.trust_scorer import TrustLevel, TrustScorer


def _r(score: float, confidence: float = 0.8, prediction: str = "real") -> ModuleResult:
    return ModuleResult(score=score, confidence=confidence, prediction=prediction)


def test_single_module_uses_full_weight():
    # When only image is present, image weight redistributes to 1.0.
    result = TrustScorer().compute_trust_score(image_result=_r(0.8))
    assert result.trust_score == pytest.approx(0.8)
    assert result.trust_level == TrustLevel.HIGH
    assert result.metadata["weights_used"] == {"image": 1.0}


def test_two_modules_weights_sum_to_one():
    out = TrustScorer().compute_trust_score(
        image_result=_r(0.9, 0.8),
        text_result=_r(0.5, 0.6),
    )
    weights = out.metadata["weights_used"]
    assert sum(weights.values()) == pytest.approx(1.0)
    # Score is a convex combination of the inputs.
    assert min(0.5, 0.9) <= out.trust_score <= max(0.5, 0.9)


def test_no_modules_raises():
    with pytest.raises(ValueError):
        TrustScorer().compute_trust_score()


def test_average_confidence_is_finite_with_zero_inputs():
    # Regression: np.mean([0.0]) used to be fine, but np.mean over a
    # zero-confidence single module produced misleading interpretations.
    out = TrustScorer().compute_trust_score(image_result=_r(0.5, 0.0))
    avg = out.metadata["average_confidence"]
    assert math.isfinite(avg)
    assert avg == pytest.approx(0.0)
