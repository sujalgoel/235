"""
SHAP and LIME explainability for text classification.

Implements:
- SHAP (SHapley Additive exPlanations) for consistent feature attribution
- LIME (Local Interpretable Model-agnostic Explanations) for local explanations
"""

from typing import Dict, Any, List, Tuple, Callable
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.logging import get_logger

logger = get_logger(__name__)


class TextExplainer:
    """
    Text explainability using SHAP and LIME.

    Provides token-level importance scores showing which words
    most influenced the model's prediction.
    """

    def __init__(self, model, tokenizer, device: str = "cpu"):
        """
        Initialize text explainer.

        Args:
            model: DistilBERT model
            tokenizer: DistilBERT tokenizer
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def explain_with_shap(self, text: str, num_samples: int = 100) -> Dict[str, Any]:
        """
        Generate SHAP explanation.

        Computes Shapley values for each token:
        φ_i = Σ_{S⊆N\{i}} [|S|!(|N|-|S|-1)!] / |N|! * [f(S∪{i}) - f(S)]

        Args:
            text: Input text
            num_samples: Number of samples for approximation

        Returns:
            Dictionary with token importances
        """
        try:
            # Tokenize
            tokens = self.tokenizer.tokenize(text)
            if not tokens:
                return {"error": "No tokens to explain"}

            # Get model prediction function
            def predict_fn(texts: List[str]) -> np.ndarray:
                """Prediction function for SHAP"""
                probs = []
                for t in texts:
                    inputs = self.tokenizer(
                        t,
                        max_length=128,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        prob = F.softmax(outputs.logits, dim=1)[0, 1].item()  # AI probability
                    probs.append(prob)

                return np.array(probs)

            # Compute approximate Shapley values
            shapley_values = self._compute_shapley_values(
                text, tokens, predict_fn, num_samples
            )

            # Get top tokens by importance
            token_importance = list(zip(tokens, shapley_values))
            token_importance.sort(key=lambda x: abs(x[1]), reverse=True)

            return {
                "tokens": tokens,
                "shapley_values": shapley_values.tolist(),
                "top_tokens": [
                    {"token": t, "importance": float(v)}
                    for t, v in token_importance[:10]
                ]
            }

        except Exception as e:
            logger.error("shap_explanation_failed", error=str(e))
            return {"error": str(e)}

    def _compute_shapley_values(
        self,
        text: str,
        tokens: List[str],
        predict_fn: Callable,
        num_samples: int
    ) -> np.ndarray:
        """
        Compute approximate Shapley values using sampling.

        Args:
            text: Original text
            tokens: List of tokens
            predict_fn: Prediction function
            num_samples: Number of samples

        Returns:
            Shapley values for each token
        """
        n_tokens = len(tokens)
        shapley_values = np.zeros(n_tokens)

        # Get baseline prediction (empty text)
        baseline_pred = predict_fn([""])[0]

        # Get full prediction
        full_pred = predict_fn([text])[0]

        # Sample coalitions and compute marginal contributions
        for _ in range(num_samples):
            # Random coalition (subset of tokens)
            coalition = np.random.choice([False, True], size=n_tokens)

            # For each token
            for i in range(n_tokens):
                if not coalition[i]:
                    # Add token i to coalition
                    coalition_with_i = coalition.copy()
                    coalition_with_i[i] = True

                    # Create texts with and without token i
                    text_without = self._create_text_from_coalition(tokens, coalition)
                    text_with = self._create_text_from_coalition(tokens, coalition_with_i)

                    # Compute marginal contribution
                    pred_without = predict_fn([text_without])[0] if text_without else baseline_pred
                    pred_with = predict_fn([text_with])[0] if text_with else baseline_pred

                    marginal = pred_with - pred_without
                    shapley_values[i] += marginal

        # Average over samples
        shapley_values /= num_samples

        return shapley_values

    def _create_text_from_coalition(self, tokens: List[str], coalition: np.ndarray) -> str:
        """
        Create text from subset of tokens.

        Args:
            tokens: All tokens
            coalition: Boolean mask of selected tokens

        Returns:
            Text with selected tokens
        """
        selected_tokens = [t for t, include in zip(tokens, coalition) if include]
        text = self.tokenizer.convert_tokens_to_string(selected_tokens)
        return text

    def explain_with_lime(self, text: str, num_samples: int = 100) -> Dict[str, Any]:
        """
        Generate LIME explanation.

        LIME creates local linear approximations:
        1. Perturb input by masking tokens
        2. Get predictions for perturbed inputs
        3. Fit weighted linear model
        4. Extract feature importance from coefficients

        Args:
            text: Input text
            num_samples: Number of perturbed samples

        Returns:
            Dictionary with token importances
        """
        try:
            # Tokenize
            tokens = self.tokenizer.tokenize(text)
            if not tokens:
                return {"error": "No tokens to explain"}

            # Get original prediction
            original_pred = self._predict_single(text)

            # Generate perturbed samples
            perturbed_samples = []
            predictions = []

            for _ in range(num_samples):
                # Random token masking
                mask = np.random.choice([0, 1], size=len(tokens), p=[0.3, 0.7])
                perturbed_text = self._mask_tokens(tokens, mask)

                if perturbed_text:
                    pred = self._predict_single(perturbed_text)
                    perturbed_samples.append(mask)
                    predictions.append(pred)

            if not perturbed_samples:
                return {"error": "No valid perturbed samples"}

            # Convert to arrays
            X = np.array(perturbed_samples)  # [num_samples, num_tokens]
            y = np.array(predictions)

            # Compute sample weights based on similarity to original
            weights = np.exp(-np.sum((X - 1) ** 2, axis=1) / (2 * len(tokens)))

            # Fit weighted linear regression
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=0.1)
            model.fit(X, y, sample_weight=weights)

            # Feature importance from coefficients
            importance = model.coef_

            # Get top tokens
            token_importance = list(zip(tokens, importance))
            token_importance.sort(key=lambda x: abs(x[1]), reverse=True)

            return {
                "tokens": tokens,
                "importance": importance.tolist(),
                "top_tokens": [
                    {"token": t, "importance": float(v)}
                    for t, v in token_importance[:10]
                ],
                "intercept": float(model.intercept_)
            }

        except Exception as e:
            logger.error("lime_explanation_failed", error=str(e))
            return {"error": str(e)}

    def _predict_single(self, text: str) -> float:
        """
        Get prediction for single text.

        Args:
            text: Input text

        Returns:
            Probability of AI class
        """
        inputs = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            prob = F.softmax(outputs.logits, dim=1)[0, 1].item()  # AI probability

        return prob

    def _mask_tokens(self, tokens: List[str], mask: np.ndarray) -> str:
        """
        Mask tokens according to binary mask.

        Args:
            tokens: List of tokens
            mask: Binary mask (1=keep, 0=remove)

        Returns:
            Text with masked tokens
        """
        selected_tokens = [t for t, m in zip(tokens, mask) if m == 1]
        if not selected_tokens:
            return ""

        text = self.tokenizer.convert_tokens_to_string(selected_tokens)
        return text
