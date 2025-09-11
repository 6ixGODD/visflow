from __future__ import annotations

import pathlib as p
import typing as t

import cv2
import PIL.Image
import torch

from visflow.data import ImageDatamodule
from visflow.helpers.gradcam import GraphCAM
from visflow.pipelines import BasePipeline
from visflow.resources.configs import TrainConfig
from visflow.resources.models import load_model_from_ckpt
from visflow.types import PathLikes
from visflow.utils import spinner


def get_colormap(
    cm: t.Literal['jet', 'turbo', 'viridis', 'inferno', 'plasma']
) -> cv2.ColormapTypes:
    colormaps = {
        'jet': cv2.COLORMAP_JET,
        'turbo': cv2.COLORMAP_TURBO,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'inferno': cv2.COLORMAP_INFERNO,
        'plasma': cv2.COLORMAP_PLASMA,
    }
    return colormaps[cm]


class GradCAMPipeline(BasePipeline):
    def __init__(
        self,
        *,
        ckpt_path: PathLikes,
        image_path: PathLikes,
        output_dir: PathLikes | None = None,
        target_layer: str | None = None,
        heatmap_only: bool = False,
        target_class: str | int | None = None,
        alpha: float = 0.5,
        colormap: t.Literal[
            'jet', 'turbo', 'viridis', 'inferno', 'plasma'
        ] = 'jet',
        eigen_smooth: bool = False,
        aug_smooth: bool = False,
        device: t.Literal['cpu', 'cuda'] | None = None,
    ):
        self.ckpt_path = p.Path(ckpt_path)
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {self.ckpt_path}"
            )
        if not self.ckpt_path.is_file():
            raise ValueError(f"Checkpoint path is not a file: {self.ckpt_path}")

        self.image_path = p.Path(image_path)
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image file not found: {self.image_path}")
        if not self.image_path.is_file():
            raise ValueError(f"Image path is not a file: {self.image_path}")

        self.output_dir = p.Path(output_dir or "./output") / 'gradcam'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_layer = target_layer
        self.heatmap_only = heatmap_only
        self.target_class = target_class
        self.alpha = alpha
        self.colormap = get_colormap(colormap)
        self.eigen_smooth = eigen_smooth
        self.aug_smooth = aug_smooth
        self.device = device

    def __call__(self) -> None:
        spinner.start('Generating Grad-CAM...')
        ckpt = torch.load(self.ckpt_path, map_location=self.device or "cpu")
        config = ckpt.get('config')
        model = load_model_from_ckpt(ckpt, map_location=self.device or "cpu")
        gradcam = GraphCAM(
            model=model,
            device=self.device,
            target_layer=self.target_layer
        )
        train_config = TrainConfig.model_validate(config)
        transform = (ImageDatamodule(train_config)
                     .val_transforms)  # Val transforms is fit for inference

        image = PIL.Image.open(self.image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(self.device or "cpu")

        if self.heatmap_only:
            gradcam.save_heatmap(
                input_tensor=input_tensor,
                target_class=self.target_class,
                eigen_smooth=self.eigen_smooth,
                aug_smooth=self.aug_smooth,
                colormap=self.colormap,
                save_path=self.output_dir / f"{self.image_path.stem}_heatmap.png"
            )
            spinner.succeed(
                f'Heatmap saved to '
                f'{self.output_dir / f"{self.image_path.stem}_heatmap.png"}'
            )
        else:
            ori_image = cv2.resize(cv2.imread(str(self.image_path)), (224, 224))
            ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB) / 255.0
            gradcam.save_cam(
                input_tensor=input_tensor,
                original_image=ori_image,
                target_class=self.target_class,
                alpha=self.alpha,
                eigen_smooth=self.eigen_smooth,
                aug_smooth=self.aug_smooth,
                colormap=self.colormap,
                save_path=self.output_dir / f"{self.image_path.stem}_cam.png"
            )
            spinner.succeed(
                f'Grad-CAM image saved to '
                f'{self.output_dir / f"{self.image_path.stem}_cam.png"}'
            )
