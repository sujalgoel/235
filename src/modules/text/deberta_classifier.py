"""
RoBERTa-Large AI text detector for ChatGPT and modern LLMs.

Uses Hello-SimpleAI's ChatGPT Detector (Hello-SimpleAI/chatgpt-detector-roberta):
- 95%+ accuracy on GPT-3.5, GPT-4
- Robust and well-tested
- Fine-tuned RoBERTa-large (355M parameters)
- More reliable than DeBERTa for production use
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from pathlib import Path

from src.modules.base import BaseModule, ModuleResult
from src.utils.logging import get_logger
from src.utils.exceptions import ModelLoadError, TextProcessingError, PredictionError

logger = get_logger(__name__)


class DeBERTaTextDetector(BaseModule):
    """
    AI-generated text detection using RoBERTa-large.

    Features:
    - 95%+ accuracy on GPT-3.5, GPT-4
    - Runs locally on Mac (no API costs)
    - Sentence-level analysis
    - Robust and production-ready
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, name="RoBERTaTextDetector")

        # Model parameters
        self.model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
        self.max_length = config.get("max_sequence_length", 512)

        # Components
        self.tokenizer = None
        self.model = None
        self.classifier_pipeline = None

    def load_model(self) -> None:
        """Load RoBERTa tokenizer and model from HuggingFace"""
        try:
            logger.info("loading_roberta_model", model=self.model_name)

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True
            )

            # Load model with proper settings for Mac
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=False
            )

            # Move to device and set eval mode
            self.model.to(self._device)
            self.model.eval()

            # Create classification pipeline for easier inference
            self.classifier_pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1 if self._device == "cpu" else 0,
                return_all_scores=True,
                top_k=None
            )

            self._is_loaded = True
            logger.info("roberta_model_loaded",
                       device=self._device,
                       params="355M",
                       max_length=self.max_length)

        except Exception as e:
            logger.error("roberta_load_failed", error=str(e))
            raise ModelLoadError(
                f"Failed to load RoBERTa model. "
                f"Make sure you have internet connection for first-time download.\n"
                f"Model: {self.model_name}\n"
                f"Error: {e}"
            )

    def preprocess(self, input_data: Any) -> Tuple[str, Dict]:
        """
        Preprocess text for DeBERTa classification.

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

            # Split into sentences for sentence-level analysis
            sentences = self._split_sentences(text)

            metadata = {
                "text_length": len(text),
                "word_count": len(text.split()),
                "sentence_count": len(sentences),
                "char_count": len(text)
            }

            return text, metadata

        except Exception as e:
            raise TextProcessingError(f"Text preprocessing failed: {e}")

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        import re

        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        return text.strip()

    def _split_sentences(self, text: str) -> list:
        """Split text into sentences for analysis"""
        import re

        # Simple sentence splitting (can be improved with nltk if needed)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

        return sentences

    def predict(self, preprocessed_data: Any) -> ModuleResult:
        """
        Classify text as human or AI-generated using DeBERTa.

        Args:
            preprocessed_data: Tuple from preprocess()

        Returns:
            ModuleResult with authenticity score and prediction
        """
        try:
            text, metadata = preprocessed_data

            logger.info("analyzing_text_with_roberta", length=len(text))

            # Use pipeline for classification
            results = self.classifier_pipeline(text, truncation=True, max_length=self.max_length)

            # Parse results (returns list of dicts with label and score)
            # RoBERTa model outputs: [{'label': 'LABEL_0' or 'LABEL_1', 'score': confidence}]
            label_scores = results[0]  # List of {label, score}

            # Find probabilities
            # Model outputs: 'Human' or 'ChatGPT' labels
            prob_human = 0.0
            prob_ai = 0.0

            for item in label_scores:
                label = item['label']
                score = item['score']

                if 'Human' in label or 'LABEL_0' in label:
                    prob_human = score
                elif 'ChatGPT' in label or 'AI' in label or 'LABEL_1' in label:
                    prob_ai = score

            # Authenticity score (probability of being human)
            score = prob_human
            confidence = max(prob_human, prob_ai)
            prediction = "human" if prob_human > prob_ai else "ai"

            # Analyze linguistic patterns
            patterns = self._analyze_patterns(text)

            # Sentence-level analysis
            sentence_analysis = self._analyze_sentences(text)

            metadata.update({
                "prob_human": float(prob_human),
                "prob_ai": float(prob_ai),
                "model_type": "RoBERTa-large (355M params)",
                "model_source": "Hello-SimpleAI/chatgpt-detector-roberta",
                "linguistic_patterns": patterns,
                "sentence_analysis": sentence_analysis
            })

            return ModuleResult(
                score=score,
                confidence=confidence,
                prediction=prediction,
                explanation={},  # Will be filled by explain()
                metadata=metadata,
                raw_output={"results": results, "text": text}
            )

        except Exception as e:
            logger.error("roberta_prediction_failed", error=str(e))
            raise PredictionError(f"RoBERTa text prediction failed: {e}")

    def _analyze_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic patterns indicative of AI generation"""
        words = text.split()
        sentences = self._split_sentences(text)

        patterns = {
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "avg_sentence_length": np.mean([len(s.split()) for s in sentences]) if sentences else 0,
            "vocabulary_diversity": len(set(words)) / len(words) if words else 0,
            "contains_ai_phrases": self._check_ai_phrases(text),
            "sentence_uniformity": self._check_sentence_uniformity(sentences)
        }

        return patterns

    def _check_ai_phrases(self, text: str) -> list:
        """Check for common AI-generated phrases"""
        ai_phrases = [
            'passionate about',
            'dedicated to',
            'innovative solutions',
            'cutting-edge',
            'leverage',
            'synergy',
            'it is important to note',
            'furthermore',
            'moreover',
            'in conclusion',
            'delve into'
        ]

        found_phrases = [phrase for phrase in ai_phrases if phrase in text.lower()]
        return found_phrases

    def _check_sentence_uniformity(self, sentences: list) -> float:
        """Calculate sentence structure uniformity (high = AI-like)"""
        if len(sentences) < 2:
            return 0.0

        lengths = [len(s.split()) for s in sentences]
        std_dev = np.std(lengths)
        mean_len = np.mean(lengths)

        # Coefficient of variation (lower = more uniform = more AI-like)
        if mean_len > 0:
            cv = std_dev / mean_len
            uniformity = max(0, 1 - cv)  # Convert to uniformity score
        else:
            uniformity = 0.0

        return float(uniformity)

    def _analyze_sentences(self, text: str) -> Dict[str, Any]:
        """Analyze individual sentences for AI probability"""
        sentences = self._split_sentences(text)

        if not sentences or len(sentences) == 0:
            return {"analyzed": False}

        # Analyze up to 5 sentences to avoid slowdown
        sample_sentences = sentences[:5]
        sentence_results = []

        for sent in sample_sentences:
            if len(sent.strip()) < 20:  # Skip very short sentences
                continue

            try:
                result = self.classifier_pipeline(sent, truncation=True, max_length=self.max_length)
                label_scores = result[0]

                ai_score = 0.0
                for item in label_scores:
                    if item['label'] == 'LABEL_1':  # AI-generated
                        ai_score = item['score']

                sentence_results.append({
                    "text_preview": sent[:80] + "..." if len(sent) > 80 else sent,
                    "ai_probability": float(ai_score),
                    "label": "AI" if ai_score > 0.5 else "Human"
                })
            except Exception as e:
                logger.debug("sentence_analysis_failed", error=str(e))
                continue

        return {
            "analyzed": True,
            "total_sentences": len(sentences),
            "analyzed_sentences": len(sentence_results),
            "sentence_scores": sentence_results,
            "avg_ai_probability": np.mean([s['ai_probability'] for s in sentence_results]) if sentence_results else 0.0
        }

    def explain(self, input_data: Any, prediction: Any) -> Dict[str, Any]:
        """Generate explanation for AI detection"""
        try:
            text = prediction.get("text", "")
            results = prediction.get("results", [])

            if not results:
                return {"error": "No results for explanation"}

            # Build explanation
            label_scores = results[0]

            explanation = {
                "model": "RoBERTa-large (ChatGPT Detector)",
                "provider": "Hello-SimpleAI",
                "accuracy": "95%+ on GPT-3.5, GPT-4",
                "class_probabilities": {},
                "ai_indicators": [],
                "key_features": []
            }

            # Extract class probabilities
            for item in label_scores:
                explanation["class_probabilities"][item['label']] = float(item['score'])

            # Identify AI indicators based on scores
            prob_ai = explanation["class_probabilities"].get('LABEL_1', 0.0)

            if prob_ai > 0.8:
                explanation["ai_indicators"] = [
                    "Very high AI probability (>80%)",
                    "Text patterns strongly match AI training data",
                    "Likely generated by GPT-4 or similar LLM"
                ]
            elif prob_ai > 0.6:
                explanation["ai_indicators"] = [
                    "High AI probability (60-80%)",
                    "Significant AI-generated content detected",
                    "May include human edits or paraphrasing"
                ]
            elif prob_ai > 0.4:
                explanation["ai_indicators"] = [
                    "Moderate AI probability (40-60%)",
                    "Mixed signals - unclear classification",
                    "Could be human-written or heavily edited AI content"
                ]
            else:
                explanation["ai_indicators"] = [
                    "Low AI probability (<40%)",
                    "Text patterns match human writing",
                    "Natural language variation detected"
                ]

            # Add linguistic features
            ai_phrases = self._check_ai_phrases(text)
            if ai_phrases:
                explanation["key_features"].append({
                    "feature": "AI-typical phrases detected",
                    "examples": ai_phrases[:3]
                })

            sentences = self._split_sentences(text)
            if sentences:
                uniformity = self._check_sentence_uniformity(sentences)
                if uniformity > 0.7:
                    explanation["key_features"].append({
                        "feature": "High sentence uniformity",
                        "score": float(uniformity)
                    })

            return explanation

        except Exception as e:
            logger.error("explanation_failed", error=str(e))
            return {"error": str(e)}
