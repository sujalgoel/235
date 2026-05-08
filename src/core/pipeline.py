"""End-to-end analysis pipeline orchestrating image + text detection."""

import time
from typing import Any, Dict, Optional

from config.base import Config
from src.modules.fusion.trust_scorer import TrustScorer
from src.utils.exceptions import RealityCheckException
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AnalysisPipeline:
    """Orchestrate ensemble image and text detection plus trust-score fusion.

    Heavy ML imports (timm, transformers, clip, cv2) are deferred to
    `initialize()` so that simply constructing the pipeline — or importing
    the FastAPI app for tests — does not require those packages.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.image_module = None
        self.text_module = None
        self.trust_scorer: Optional[TrustScorer] = None
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return

        # Defer the heavy imports until first use.
        from src.modules.image.ensemble_detector import EnsembleImageDetector
        from src.modules.text.ensemble_text_detector import EnsembleTextDetector

        logger.info("initializing_pipeline")
        try:
            self.image_module = EnsembleImageDetector(self.config.MODEL.__dict__)
            self.text_module = EnsembleTextDetector(self.config.MODEL.__dict__)
            self.trust_scorer = TrustScorer(self.config.FUSION)
            self.image_module.load_model()
            self.text_module.load_model()
            self._initialized = True
            logger.info("pipeline_initialized")
        except Exception as e:
            logger.error("pipeline_initialization_failed", error=str(e))
            raise RealityCheckException(f"Failed to initialize pipeline: {e}")

    def analyze_profile(
        self,
        image_path: Optional[str] = None,
        text: Optional[str] = None,
        profile_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self._initialized:
            self.initialize()

        start_time = time.time()
        logger.info(
            "analyzing_profile",
            profile_id=profile_id,
            has_image=image_path is not None,
            has_text=text is not None,
        )

        image_result = None
        text_result = None

        if image_path:
            try:
                image_result = self.image_module(image_path, explain=True)
                logger.info("image_analysis_complete", score=image_result.score)
            except Exception as e:
                logger.error("image_analysis_failed", error=str(e))

        if text:
            try:
                text_result = self.text_module(text, explain=True)
                logger.info("text_analysis_complete", score=text_result.score)
            except Exception as e:
                logger.error("text_analysis_failed", error=str(e))

        if not (image_result or text_result):
            raise RealityCheckException("No successful analysis from any module")

        trust_result = self.trust_scorer.compute_trust_score(
            image_result=image_result,
            text_result=text_result,
        )
        processing_time = (time.time() - start_time) * 1000

        return self._envelope(
            profile_id=profile_id,
            image_result=image_result,
            text_result=text_result,
            trust_result=trust_result,
            processing_time=processing_time,
        )

    def analyze_image_only(self, image_path: str) -> Dict[str, Any]:
        if not self._initialized:
            self.initialize()

        start_time = time.time()
        result = self.image_module(image_path, explain=True)
        trust_result = self.trust_scorer.compute_trust_score(image_result=result)

        return self._envelope(
            profile_id=None,
            image_result=result,
            text_result=None,
            trust_result=trust_result,
            processing_time=(time.time() - start_time) * 1000,
        )

    def analyze_text_only(self, text: str) -> Dict[str, Any]:
        if not self._initialized:
            self.initialize()

        start_time = time.time()
        result = self.text_module(text, explain=True)
        trust_result = self.trust_scorer.compute_trust_score(text_result=result)

        return self._envelope(
            profile_id=None,
            image_result=None,
            text_result=result,
            trust_result=trust_result,
            processing_time=(time.time() - start_time) * 1000,
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "modules": {
                "image": self.image_module.is_loaded() if self.image_module else False,
                "text": self.text_module.is_loaded() if self.text_module else False,
            },
            "device": self.config.MODEL.device,
        }

    def _envelope(self, *, profile_id, image_result, text_result, trust_result, processing_time):
        return {
            "profile_id": profile_id or "unknown",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "image_analysis": image_result.to_dict() if image_result else None,
            "text_analysis": text_result.to_dict() if text_result else None,
            "trust_score_result": trust_result.to_dict(),
            "processing_time_ms": processing_time,
        }
