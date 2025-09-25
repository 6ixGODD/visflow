from __future__ import annotations

import argparse
import os
import typing as t

from visflow._cli.args import BaseArgs
from visflow.pipelines.gradcam import GradCAMPipeline

if t.TYPE_CHECKING:
    from argparse import _SubParsersAction


class Args(BaseArgs):
    ckpt_path: str
    image_path: str
    output_dir: str = "./output"
    target_layer: str | None = None
    heatmap_only: bool = False
    target_class: int | str | None = None
    alpha: float = 0.5
    colormap: t.Literal["jet", "turbo", "viridis", "inferno", "plasma"] = "jet"
    eigen_smooth: bool = False
    aug_smooth: bool = False
    device: t.Literal["cpu", "cuda"] | None = "cuda"
    verbose: bool = False

    def run(self) -> None:
        if self.verbose:
            os.environ["VF_VERBOSE"] = "1"
        pipeline = GradCAMPipeline(ckpt_path=self.ckpt_path,
                                   image_path=self.image_path,
                                   output_dir=self.output_dir,
                                   target_layer=self.target_layer,
                                   heatmap_only=self.heatmap_only,
                                   target_class=self.target_class,
                                   alpha=self.alpha,
                                   colormap=self.colormap,
                                   eigen_smooth=self.eigen_smooth,
                                   aug_smooth=self.aug_smooth,
                                   device=self.device)
        pipeline()

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--ckpt-path",
                            "-k",
                            type=str,
                            required=True,
                            help="Path to the model checkpoint file.")
        parser.add_argument("--image-path",
                            "-i",
                            type=str,
                            required=True,
                            help="Path to the input image file.")
        parser.add_argument("--output-dir",
                            "-o",
                            type=str,
                            default="./output",
                            help="Directory to save the output visualizations. (default: "
                            "%(default)s)")
        parser.add_argument("--target-layer",
                            "-l",
                            type=str,
                            default=None,
                            help="Name of the target convolutional layer to visualize. If not "
                            "specified, the last convolutional layer will be used.")
        parser.add_argument("--heatmap-only",
                            action="store_true",
                            help="If set, only the heatmap will be saved without overlaying "
                            "it on the original image. (default: %(default)s)")
        parser.add_argument("--target-class",
                            "-t",
                            type=str,
                            default=None,
                            help="Target class for which to generate the Grad-CAM. Can be the "
                            "class index or class name. If not specified, the predicted "
                            "class will be used.")
        parser.add_argument("--alpha",
                            "-a",
                            type=float,
                            default=0.5,
                            help="Transparency factor for overlaying the heatmap on the "
                            "original image. Value should be between 0 and 1. (default: "
                            "%(default)s)")
        parser.add_argument("--colormap",
                            "-c",
                            type=str,
                            choices=["jet", "turbo", "viridis", "inferno", "plasma"],
                            default="jet",
                            help="Colormap to use for the heatmap. (default: %(default)s)")
        parser.add_argument("--eigen-smooth",
                            action="store_true",
                            help="If set, apply eigen-smoothing to the Grad-CAM. (default: "
                            "%(default)s)")
        parser.add_argument("--aug-smooth",
                            action="store_true",
                            help="If set, apply augmented smoothing to the Grad-CAM. (default: "
                            "%(default)s)")
        parser.add_argument("--device",
                            "-d",
                            type=str,
                            choices=["cpu", "cuda"],
                            default=None,
                            help="Device to run the Grad-CAM computation on. If not specified, "
                            'will use "cuda" if available, otherwise "cpu". (default: '
                            "%(default)s)")
        parser.add_argument("--verbose",
                            "-v",
                            action="store_true",
                            help="If set, enable verbose logging. (default: %(default)s)")


def register(subparser: _SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparser.add_parser("gradcam", help="Visualize model predictions using Grad-CAM")
    Args.add_args(parser)
    parser.set_defaults(func=Args.func)
