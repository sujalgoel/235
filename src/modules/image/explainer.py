"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for visual explanations.

Implements the Grad-CAM algorithm from the paper:
"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


class GradCAMExplainer:
    """
    Grad-CAM explainability for CNNs.

    Generates visual explanations showing which regions of the image
    most influenced the model's decision.

    Implements Equation 2 from the paper:
    L^c_Grad-CAM = ReLU(Σ_k α^c_k A^k)

    where α^c_k are the importance weights for feature map k.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM explainer.

        Args:
            model: PyTorch model
            target_layer: Target convolutional layer (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on target layer"""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class for explanation (uses predicted if None)

        Returns:
            Heatmap as numpy array [H, W] with values in [0, 1]
        """
        self.model.eval()
        input_tensor.requires_grad = True

        # Forward pass
        output = self.model(input_tensor)

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        class_score = output[0, target_class]
        class_score.backward()

        # Get gradients and activations
        gradients = self.gradients[0].cpu().numpy()  # [C, H, W]
        activations = self.activations[0].cpu().numpy()  # [C, H, W]

        # Compute importance weights (α^c_k)
        # α^c_k = (1/Z) * Σ_i Σ_j ∂y^c / ∂A^k_ij
        weights = np.mean(gradients, axis=(1, 2))  # [C]

        # Weighted combination of activation maps
        # L^c = Σ_k α^c_k * A^k
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU
        cam = np.maximum(cam, 0)

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input size
        input_size = input_tensor.shape[2:]  # [H, W]
        cam = cv2.resize(cam, (input_size[1], input_size[0]))

        return cam

    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original image.

        Args:
            image: Original image [H, W, C] (RGB)
            heatmap: Grad-CAM heatmap [H, W]
            colormap: OpenCV colormap
            alpha: Overlay transparency

        Returns:
            Image with heatmap overlay [H, W, C] (RGB)
        """
        # Ensure heatmap matches image size
        if heatmap.shape != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        # Convert heatmap to uint8 [0, 255]
        heatmap_uint8 = np.uint8(255 * heatmap)

        # Apply colormap
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

        # Convert BGR to RGB
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = np.uint8(image)

        # Overlay heatmap on image
        overlay = cv2.addWeighted(image, 1 - alpha, colored_heatmap, alpha, 0)

        return overlay

    def save_visualization(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        output_path: str
    ):
        """
        Save Grad-CAM visualization to file.

        Args:
            image: Original image
            heatmap: Grad-CAM heatmap
            output_path: Path to save visualization
        """
        overlay = self.overlay_heatmap(image, heatmap)

        # Convert RGB to BGR for saving
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, overlay_bgr)
