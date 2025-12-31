"""
DistilBERT text authenticity classifier with SHAP/LIME explainability.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    DistilBertConfig
)
from pathlib import Path

from src.modules.base import BaseModule, ModuleResult
from src.modules.text.explainer import TextExplainer
from src.utils.logging import get_logger
from src.utils.exceptions import ModelLoadError, TextProcessingError, PredictionError

logger = get_logger(__name__)


class TextAuthenticityModule(BaseModule):
    """
    Text authenticity detection module using DistilBERT.

    Pipeline:
    1. Text preprocessing and tokenization
    2. DistilBERT classification (human vs AI-generated)
    3. SHAP/LIME explanation generation
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, name="TextAuthenticityModule")

        # Model parameters
        self.model_name = config.get("text_model_name", "distilbert-base-uncased")
        self.max_length = config.get("max_sequence_length", 128)

        # Components
        self.tokenizer = None
        self.model = None
        self.explainer = None

    def load_model(self) -> None:
        """Load DistilBERT tokenizer and model"""
        try:
            logger.info("loading_text_models")

            # Load tokenizer
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

            # Load model
            model_path = self.config.get("text_model_path")

            if model_path and Path(model_path).exists():
                # Load fine-tuned model
                logger.info("loading_finetuned_model", path=model_path)
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=2
                )
            else:
                # Initialize with pretrained weights (not fine-tuned)
                logger.warning("loading_base_model_not_finetuned")
                config = DistilBertConfig.from_pretrained(self.model_name)
                config.num_labels = 2
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    self.model_name,
                    config=config
                )

            # Move to device and set eval mode
            self.model.to(self._device)
            self.model.eval()

            # Initialize explainer
            self.explainer = TextExplainer(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self._device
            )

            self._is_loaded = True
            logger.info("text_models_loaded", device=self._device)

        except Exception as e:
            raise ModelLoadError(f"Failed to load text models: {e}")

    def preprocess(self, input_data: Any) -> Tuple[Dict[str, torch.Tensor], str]:
        """
        Preprocess text for classification.

        Args:
            input_data: Text string or dict with 'text' key

        Returns:
            Tuple of (tokenized_inputs, cleaned_text)
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

            # Tokenize
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            return inputs, text

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

        # Remove URLs (optional)
        import re
        text = re.sub(r'http\S+', '', text)

        # Remove mentions/hashtags (optional for social media)
        # text = re.sub(r'[@#]\w+', '', text)

        return text.strip()

    def predict(self, preprocessed_data: Any) -> ModuleResult:
        """
        Classify text as human or AI-generated.

        Args:
            preprocessed_data: Tuple from preprocess()

        Returns:
            ModuleResult with authenticity score and prediction
        """
        try:
            inputs, text = preprocessed_data

            # Move to device
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)

            # Extract probabilities (0=human, 1=AI)
            prob_human = float(probs[0, 0])
            prob_ai = float(probs[0, 1])

            # Authenticity score (probability of being human)
            score = prob_human
            confidence = max(prob_human, prob_ai)
            prediction = "human" if prob_human > prob_ai else "ai"

            # Analyze linguistic patterns
            patterns = self._analyze_patterns(text)

            metadata = {
                "prob_human": prob_human,
                "prob_ai": prob_ai,
                "text_length": len(text),
                "word_count": len(text.split()),
                "model_type": "DistilBERT",
                "linguistic_patterns": patterns
            }

            return ModuleResult(
                score=score,
                confidence=confidence,
                prediction=prediction,
                explanation={},  # Will be filled by explain()
                metadata=metadata,
                raw_output={"logits": logits, "probs": probs, "text": text, "inputs": inputs}
            )

        except Exception as e:
            raise PredictionError(f"Text prediction failed: {e}")

    def _analyze_patterns(self, text: str) -> Dict[str, Any]:
        """
        Analyze linguistic patterns that may indicate AI generation.

        Args:
            text: Input text

        Returns:
            Dictionary of detected patterns
        """
        words = text.split()
        sentences = text.split('.')

        patterns = {
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "avg_sentence_length": np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0,
            "contains_formal_words": any(word in text.lower() for word in [
                'passionate', 'dedicated', 'innovative', 'enthusiastic', 'leverage'
            ]),
            "perfect_grammar": not any(char in text for char in ['...', '!!', 'lol', 'btw']),
            "sentence_count": len([s for s in sentences if s.strip()])
        }

        return patterns

    def explain(self, input_data: Any, prediction: Any) -> Dict[str, Any]:
        """
        Generate SHAP/LIME explanations.

        Args:
            input_data: Original input text
            prediction: Raw model output

        Returns:
            Dictionary with token importance and patterns
        """
        try:
            text = prediction.get("text")
            if not text:
                return {"error": "No text for explanation"}

            # Generate SHAP explanation
            shap_explanation = self.explainer.explain_with_shap(text)

            # Generate LIME explanation
            lime_explanation = self.explainer.explain_with_lime(text)

            # Identify AI indicators
            ai_indicators = self._identify_ai_indicators(text, shap_explanation)

            return {
                "shap": shap_explanation,
                "lime": lime_explanation,
                "ai_indicators": ai_indicators,
                "top_features": shap_explanation.get("top_tokens", [])[:5]
            }

        except Exception as e:
            logger.error("explanation_failed", error=str(e))
            return {"error": str(e)}

    def _identify_ai_indicators(
        self,
        text: str,
        shap_explanation: Dict[str, Any]
    ) -> list:
        """
        Identify linguistic patterns indicative of AI generation.

        Args:
            text: Input text
            shap_explanation: SHAP analysis results

        Returns:
            List of identified AI indicators
        """
        indicators = []

        # Check for formal language
        formal_words = ['passionate', 'dedicated', 'innovative', 'enthusiastic']
        if any(word in text.lower() for word in formal_words):
            indicators.append("Excessive use of formal vocabulary")

        # Check for perfect grammar (no typos, colloquialisms)
        if not any(char in text for char in ['...', '!!', 'lol', 'btw', 'omg']):
            indicators.append("Overly perfect grammar and structure")

        # Check sentence structure uniformity
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            lengths = [len(s.split()) for s in sentences]
            if len(set(lengths)) < len(lengths) / 2:
                indicators.append("Uniform sentence structure")

        # Check for lack of personal anecdotes
        personal_indicators = ['i', 'my', 'me', 'we', 'our']
        if not any(word in text.lower().split() for word in personal_indicators):
            indicators.append("Lack of personal pronouns and anecdotes")

        if not indicators:
            indicators.append("No strong AI indicators detected")

        return indicators
