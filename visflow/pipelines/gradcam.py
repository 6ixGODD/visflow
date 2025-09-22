from __future__ import annotations

import pathlib as p
import typing as t

import cv2
import PIL.Image
import torch

from visflow.data import ImageDatamodule
from visflow.helpers.gradcam import GraphCAM
from visflow.pipelines import BasePipeline
from visflow.resources.config import TrainConfig
from visflow.resources.models import load_model_from_ckpt
from visflow.types import Checkpoint, PathLikes
from visflow.utils import spinner


def get_colormap(
    cm: t.Literal["jet", "turbo", "viridis", "inferno", "plasma"],
) -> cv2.ColormapTypes:
    colormaps = {
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "inferno": cv2.COLORMAP_INFERNO,
        "plasma": cv2.COLORMAP_PLASMA,
    }
    return colormaps[cm]


class GradCAMPipeline(BasePipeline):
    SUPPORTED_EXTENSIONS: t.ClassVar[t.Set[str]] = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
    }

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
        colormap: t.Literal["jet", "turbo", "viridis", "inferno", "plasma"] = "jet",
        eigen_smooth: bool = False,
        aug_smooth: bool = False,
        device: t.Literal["cpu", "cuda"] | None = None,
    ):
        self._completed = False
        self.ckpt_path = p.Path(ckpt_path)
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.ckpt_path}")
        if not self.ckpt_path.is_file():
            raise ValueError(f"Checkpoint path is not a file: {self.ckpt_path}")

        self.image_path = p.Path(image_path)
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image path not found: {self.image_path}")

        if self.image_path.is_file():
            if self.image_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported image format: {self.image_path.suffix}")
            self.image_files = [self.image_path]
        elif self.image_path.is_dir():
            self.image_files = self._find_image(self.image_path)
            if not self.image_files:
                raise ValueError(
                    f"No supported image files found in directory: "
                    f"{self.image_path}"
                )
        else:
            raise ValueError(
                f"Image path is neither a file nor a directory: "
                f""
                f"{self.image_path}"
            )

        self.output_dir = p.Path(output_dir or "./output") / "gradcam"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_layer = target_layer
        self.heatmap_only = heatmap_only
        self.target_class = target_class
        self.alpha = alpha
        self.colormap = get_colormap(colormap)
        self.eigen_smooth = eigen_smooth
        self.aug_smooth = aug_smooth
        self.device = device

    def _find_image(self, directory: p.Path) -> list[p.Path]:
        image_files = []
        for file_path in directory.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ):
                image_files.append(file_path)
        return sorted(image_files)

    def _step(
        self,
        image_file: p.Path,
        gradcam: GraphCAM,
        transform: t.Any,
        current_idx: int,
        total_count: int,
        target_class: int | None = None,
    ) -> None:
        progress_info = f"({current_idx}/{total_count}) {image_file.name}"
        spinner.text = f"Generating Grad-CAM... {progress_info}"

        image = PIL.Image.open(image_file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(self.device or "cpu")

        relative_path = (
            image_file.relative_to(self.image_path)
            if self.image_path.is_dir()
            else image_file
        )
        output_subdir = self.output_dir / relative_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)

        if self.heatmap_only:
            save_path = output_subdir / f"{image_file.stem}_heatmap.png"
            gradcam.save_heatmap(
                input_tensor=input_tensor,
                target_class=target_class,
                eigen_smooth=self.eigen_smooth,
                aug_smooth=self.aug_smooth,
                colormap=self.colormap,
                save_path=save_path,
            )
        else:
            save_path = output_subdir / f"{image_file.stem}_cam.png"
            ori_image = (
                cv2.cvtColor(
                    cv2.resize(cv2.imread(str(image_file)), (224, 224)),
                    cv2.COLOR_BGR2RGB,
                )
                / 255.0
            )
            gradcam.save_cam(
                input_tensor=input_tensor,
                original_image=ori_image,
                target_class=target_class,
                alpha=self.alpha,
                eigen_smooth=self.eigen_smooth,
                aug_smooth=self.aug_smooth,
                colormap=self.colormap,
                save_path=save_path,
            )

    def __call__(self) -> None:
        total_images = len(self.image_files)

        if total_images == 1:
            spinner.start("Generating Grad-CAM...")
        else:
            spinner.start(f"Generating Grad-CAM for {total_images} images...")

        ckpt: Checkpoint = torch.load(self.ckpt_path, map_location=self.device or "cpu")

        target_class: int | None
        if isinstance(self.target_class, str):
            target_class = ckpt["class_to_idx"][self.target_class]
        else:
            target_class = self.target_class
        config = ckpt.get("config")
        model = load_model_from_ckpt(ckpt)
        gradcam = GraphCAM(
            model=model, device=self.device, target_layer=self.target_layer
        )
        train_config = TrainConfig.model_validate(config)
        transform = ImageDatamodule(
            train_config
        ).val_transforms  # Val transforms is fit for inference

        for idx, image_file in enumerate(self.image_files, 1):
            try:
                self._step(
                    image_file, gradcam, transform, idx, total_images, target_class
                )
                spinner.info(f"Processed {image_file} ({idx}/{total_images})")
            except Exception as e:
                spinner.warn(f"Failed to process {image_file}: {e}")
                continue

        if total_images == 1:
            output_type = "heatmap" if self.heatmap_only else "Grad-CAM image"
            spinner.succeed(f"{output_type} saved to {self.output_dir}")
        else:
            output_type = "heatmaps" if self.heatmap_only else "Grad-CAM images"
            spinner.succeed(f"{total_images} {output_type} saved to {self.output_dir}")

        self._completed = True
