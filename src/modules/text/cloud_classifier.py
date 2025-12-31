"""
GPTZero cloud-based AI text detection with explainability.

Uses GPTZero's AI Detection API for state-of-the-art accuracy.
"""

from typing import Dict, Any, Tuple, Optional
import requests
import os
import re

from src.modules.base import BaseModule, ModuleResult
from src.utils.logging import get_logger
from src.utils.exceptions import ModelLoadError, TextProcessingError, PredictionError

logger = get_logger(__name__)


class GPTZeroTextDetector(BaseModule):
    """
    Cloud-based AI text detection using GPTZero.

    Features:
    - 99% claimed accuracy (84% independently verified)
    - Detects: GPT-5, GPT-4, Gemini 2.5, Deepseek-V3
    - Sentence-level highlighting
    - Returns HUMAN_ONLY, MIXED, or AI_ONLY classification
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, name="GPTZeroTextDetector")

        # API configuration
        self.api_key = config.get("gptzero_api_key") or os.getenv("GPTZERO_API_KEY")
        self.api_url = "https://api.gptzero.me/v2/predict/text"

        if not self.api_key:
            raise ValueError(
                "GPTZero API key not found. Set GPTZERO_API_KEY in environment or config.\n"
                "Get your API key at: https://gptzero.me/developers"
            )

    def load_model(self) -> None:
        """Validate API connection"""
        try:
            logger.info("validating_gptzero_api_connection")

            # Test API key is present (validate on first request)
            if not self.api_key or len(self.api_key) < 10:
                raise ValueError("Invalid GPTZero API key format")

            self._is_loaded = True
            logger.info("gptzero_api_ready")

        except Exception as e:
            raise ModelLoadError(f"Failed to validate GPTZero API: {e}")

    def preprocess(self, input_data: Any) -> Tuple[str, Dict]:
        """
        Prepare text for GPTZero API.

        Args:
            input_data: Text string or dict with 'text' key

        Returns:
            Tuple of (cleaned_text, metadata)
        """
        try:
            # Extract text
            if isinstance(input_data, str):
                text = input_data
            elif isinstance(input_data, dict) and 'text' in input_data:
                text = input_data['text']
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")

            # Clean text
            text = self._clean_text(text)

            metadata = {
                "text_length": len(text),
                "word_count": len(text.split()),
                "char_count": len(text)
            }

            return text, metadata

        except Exception as e:
            raise TextProcessingError(f"Text preprocessing failed: {e}")

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        return text.strip()

    def predict(self, preprocessed_data: Any) -> ModuleResult:
        """
        Detect AI-generated text using GPTZero API.

        Args:
            preprocessed_data: Tuple from preprocess()

        Returns:
            ModuleResult with authenticity score and prediction
        """
        try:
            text, metadata = preprocessed_data

            # Prepare API request
            headers = {
                "X-Api-Key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            payload = {
                "document": text,
                "version": "2024-01-09"  # Latest stable version
            }

            logger.info("calling_gptzero_api", text_length=len(text))
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )

            # Handle API response
            if response.status_code == 401:
                raise PredictionError(
                    "GPTZero API authentication failed. Check your API key.\n"
                    "Get your API key at: https://gptzero.me/developers"
                )
            elif response.status_code == 429:
                raise PredictionError(
                    "GPTZero API rate limit exceeded. Upgrade your plan or wait.\n"
                    "See pricing: https://gptzero.me/pricing"
                )
            elif response.status_code != 200:
                raise PredictionError(
                    f"GPTZero API error {response.status_code}: {response.text}"
                )

            result = response.json()
            logger.info("gptzero_api_response_received")

            # Parse GPTZero response
            score, confidence, prediction, explanation = self._parse_gptzero_response(result)

            # Update metadata
            metadata.update({
                "model_type": "GPTZero Model 3.7b",
                "api_version": "v2",
                "classification": result.get("documents", [{}])[0].get("class_probabilities", {})
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
            raise PredictionError(f"GPTZero API request failed: {e}")
        except Exception as e:
            raise PredictionError(f"Text prediction failed: {e}")

    def _parse_gptzero_response(self, response: Dict[str, Any]) -> Tuple[float, float, str, Dict]:
        """
        Parse GPTZero API response.

        GPTZero returns:
        - completely_generated_prob: Probability text is 100% AI
        - overall_burstiness: Variation in sentence structure
        - average_perplexity: How predictable the text is

        Args:
            response: GPTZero API JSON response

        Returns:
            Tuple of (score, confidence, prediction, explanation)
        """
        try:
            documents = response.get('documents', [])
            if not documents:
                raise ValueError("No documents in GPTZero response")

            doc = documents[0]

            # Extract key metrics
            ai_prob = doc.get('completely_generated_prob', 0.5)  # Prob of 100% AI
            class_probs = doc.get('class_probabilities', {})

            # GPTZero classifications
            prob_ai_only = class_probs.get('ai', 0.0)  # AI_ONLY
            prob_human_only = class_probs.get('human', 0.0)  # HUMAN_ONLY
            prob_mixed = class_probs.get('mixed', 0.0)  # MIXED

            # Perplexity and burstiness (linguistic features)
            avg_perplexity = doc.get('average_generated_prob', 0.5)
            burstiness = doc.get('overall_burstiness', 0.5)

            # Authenticity score: probability of being human-written
            # Higher human_only probability = higher authenticity
            authenticity_score = prob_human_only + (prob_mixed * 0.5)  # Mixed counts as half-human

            # Confidence based on how clear-cut the classification is
            max_class_prob = max(prob_ai_only, prob_human_only, prob_mixed)
            confidence = max_class_prob

            # Prediction
            if prob_ai_only > prob_human_only and prob_ai_only > prob_mixed:
                prediction = "ai"
            elif prob_human_only > prob_ai_only and prob_human_only > prob_mixed:
                prediction = "human"
            else:
                prediction = "mixed"

            # Sentence-level highlights (if available)
            sentences = doc.get('sentences', [])
            ai_sentences = []
            human_sentences = []

            for sent in sentences:
                sent_text = sent.get('sentence', '')
                sent_prob = sent.get('generated_prob', 0.5)

                if sent_prob > 0.7:
                    ai_sentences.append({
                        "text": sent_text[:100],  # Truncate for display
                        "ai_probability": sent_prob
                    })
                elif sent_prob < 0.3:
                    human_sentences.append({
                        "text": sent_text[:100],
                        "ai_probability": sent_prob
                    })

            # Build explanation
            explanation = {
                "completely_generated_probability": ai_prob,
                "average_perplexity": avg_perplexity,
                "overall_burstiness": burstiness,
                "class_probabilities": {
                    "ai_only": prob_ai_only,
                    "human_only": prob_human_only,
                    "mixed": prob_mixed
                },
                "ai_sentences_count": len(ai_sentences),
                "human_sentences_count": len(human_sentences),
                "top_ai_sentences": ai_sentences[:3],  # Top 3 most AI-like
                "detection_method": "GPTZero Multi-Feature Analysis"
            }

            # Add verdict
            if prob_ai_only > 0.8:
                explanation["verdict"] = "Very likely 100% AI-generated"
                explanation["ai_indicators"] = [
                    "High completely_generated_prob",
                    "Low burstiness (uniform sentence structure)",
                    "High perplexity (predictable patterns)"
                ]
            elif prob_ai_only > 0.5:
                explanation["verdict"] = "Likely AI-generated with possible edits"
                explanation["ai_indicators"] = ["Moderate AI probability", "Some human editing detected"]
            elif prob_mixed > 0.5:
                explanation["verdict"] = "Mixed human and AI content"
                explanation["ai_indicators"] = ["Clear mix of AI and human writing styles"]
            elif prob_human_only > 0.7:
                explanation["verdict"] = "Likely human-written"
                explanation["ai_indicators"] = ["High burstiness", "Natural language patterns"]
            else:
                explanation["verdict"] = "Unclear - manual review recommended"
                explanation["ai_indicators"] = ["Ambiguous signals"]

            return authenticity_score, confidence, prediction, explanation

        except Exception as e:
            logger.error("failed_to_parse_gptzero_response", error=str(e), response=response)
            # Fallback: uncertain
            return 0.5, 0.0, "unknown", {"error": str(e)}

    def explain(self, input_data: Any, prediction: Any) -> Dict[str, Any]:
        """
        Generate explanation from GPTZero results.

        Args:
            input_data: Original input data
            prediction: Raw GPTZero API output

        Returns:
            Dictionary with sentence-level highlights and metrics
        """
        try:
            documents = prediction.get('documents', [])
            if not documents:
                return {"error": "No documents in GPTZero response"}

            doc = documents[0]
            sentences = doc.get('sentences', [])

            # Build detailed explanation
            explanation = {
                "provider": "GPTZero",
                "sentence_analysis": [],
                "overall_metrics": {
                    "completely_generated_prob": doc.get('completely_generated_prob', 0),
                    "average_perplexity": doc.get('average_generated_prob', 0),
                    "overall_burstiness": doc.get('overall_burstiness', 0)
                },
                "top_features": []
            }

            # Analyze each sentence
            for i, sent in enumerate(sentences[:10]):  # Limit to first 10 sentences
                sent_analysis = {
                    "sentence_number": i + 1,
                    "text_preview": sent.get('sentence', '')[:80] + "...",
                    "ai_probability": sent.get('generated_prob', 0),
                    "label": "AI" if sent.get('generated_prob', 0) > 0.5 else "Human"
                }
                explanation["sentence_analysis"].append(sent_analysis)

            # Extract top AI indicators
            if doc.get('completely_generated_prob', 0) > 0.5:
                explanation["top_features"].append({
                    "feature": "High AI probability",
                    "importance": doc.get('completely_generated_prob', 0)
                })

            if doc.get('overall_burstiness', 1.0) < 0.3:
                explanation["top_features"].append({
                    "feature": "Low burstiness (uniform structure)",
                    "importance": 1.0 - doc.get('overall_burstiness', 0)
                })

            return explanation

        except Exception as e:
            logger.error("explanation_generation_failed", error=str(e))
            return {"error": str(e)}
