from __future__ import annotations

import pathlib as p
import typing as t

import cv2
import numpy as np
import pytorch_grad_cam as gradcam
import torch
import torch.nn as nn
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from visflow.resources.models import BaseClassifier, TorchVisionClassifier
from visflow.types import PathLikes


class GraphCAM:

    def __init__(self,
                 model: BaseClassifier,
                 device: t.Literal["cpu", "cuda"] | None = None,
                 target_layer: str | None = None):
        self.model = model
        self.device = torch.device(str(device) or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()

        self.target_layer = self._parse_target_layer(target_layer)

        self.gcam = gradcam.GradCAM(model=self.model, target_layers=[self.target_layer])

    def _parse_target_layer(self, target_layer: str | None) -> nn.Module:
        if target_layer is None:
            try:
                return self.model.gradcam_layer
            except NotImplementedError as e:
                raise ValueError(f"Model {type(self.model).__name__} does not implement "
                                 f"gradcam_layer() "
                                 f"and no target_layer was provided. {e}")

        # Parse target layer dynamically
        try:
            # For TorchVision models, prepend "backbone."
            if isinstance(self.model, TorchVisionClassifier):
                layer_path = f"backbone.{target_layer}"
            else:
                layer_path = target_layer

            # Use eval to dynamically access the layer
            # Note: This is safe here as we control the input and model
            # structure
            layer = eval(f"self.model.{layer_path}")

            if not isinstance(layer, nn.Module):
                raise ValueError(f"Target layer '{target_layer}' is not a nn.Module")

            return layer

        except (AttributeError, SyntaxError, NameError) as e:
            raise ValueError(f"Cannot access target layer '{target_layer}' in model "
                             f"{type(self.model).__name__}: {e}")

    def cam(self,
            input_tensor: torch.Tensor,
            target_class: int | None = None,
            eigen_smooth: bool = False,
            aug_smooth: bool = False) -> np.ndarray:
        """
        Generate GradCAM heatmap for input tensor.

        Args:
            input_tensor: Input tensor of shape (batch_size, channels,
                height, width)
            target_class: Target class index. If None, uses predicted class
            eigen_smooth: Whether to use eigen smoothing
            aug_smooth: Whether to use augmentation smoothing

        Returns:
            GradCAM heatmap as numpy array
        """
        input_tensor = input_tensor.to(self.device)

        # Prepare targets
        targets = None
        if target_class is not None:
            targets = [ClassifierOutputTarget(target_class)]

        # Generate GradCAM
        grayscale_cam = self.gcam(input_tensor=input_tensor,
                                  targets=targets,
                                  eigen_smooth=eigen_smooth,
                                  aug_smooth=aug_smooth)

        return grayscale_cam

    def save_cam(self,
                 input_tensor: torch.Tensor,
                 original_image: np.ndarray,
                 save_path: PathLikes,
                 target_class: int | None = None,
                 alpha: float = 0.4,
                 colormap: cv2.ColormapTypes = cv2.COLORMAP_JET,
                 eigen_smooth: bool = False,
                 aug_smooth: bool = False) -> None:
        """
        Generate and save GradCAM visualization overlaid on original image.

        Args:
            input_tensor: Input tensor for the model
            original_image: Original image as numpy array (RGB, 0-1 range)
            save_path: Path to save the visualization
            target_class: Target class index. If None, uses predicted class
            alpha: Transparency for overlay (0-1)
            colormap: OpenCV colormap for heatmap
            eigen_smooth: Whether to use eigen smoothing
            aug_smooth: Whether to use augmentation smoothing
        """
        # Generate GradCAM
        grayscale_cam = self.cam(input_tensor=input_tensor,
                                 target_class=target_class,
                                 eigen_smooth=eigen_smooth,
                                 aug_smooth=aug_smooth)

        # Take the first image if batch
        if len(grayscale_cam.shape) > 2:
            grayscale_cam = grayscale_cam[0]

        # Create visualization
        visualization = show_cam_on_image(img=original_image,
                                          mask=grayscale_cam,
                                          use_rgb=True,
                                          colormap=colormap,
                                          image_weight=1 - alpha)

        # Save visualization
        save_path = p.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert RGB to BGR for OpenCV
        visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), visualization_bgr)

    def save_heatmap(self,
                     input_tensor: torch.Tensor,
                     save_path: PathLikes,
                     target_class: int | None = None,
                     colormap: cv2.ColormapTypes = cv2.COLORMAP_JET,
                     eigen_smooth: bool = False,
                     aug_smooth: bool = False) -> None:
        """
        Generate and save only the GradCAM heatmap without overlay.

        Args:
            input_tensor: Input tensor for the model
            save_path: Path to save the heatmap
            target_class: Target class index. If None, uses predicted class
            colormap: OpenCV colormap for heatmap
            eigen_smooth: Whether to use eigen smoothing
            aug_smooth: Whether to use augmentation smoothing
        """
        # Generate GradCAM
        grayscale_cam = self.cam(
            input_tensor=input_tensor,
            target_class=target_class,
            eigen_smooth=eigen_smooth,
            aug_smooth=aug_smooth,
        )

        # Take the first image if batch
        if len(grayscale_cam.shape) > 2:
            grayscale_cam = grayscale_cam[0]

        # Apply colormap
        heatmap = cv2.applyColorMap((255 * grayscale_cam).astype(np.uint8), colormap)

        # Save heatmap
        save_path = p.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), heatmap)
