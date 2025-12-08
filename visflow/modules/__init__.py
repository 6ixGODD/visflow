from __future__ import annotations

import abc
import importlib
import os
import pathlib
import pkgutil
import typing as t

import torch
import torch.nn as nn

from visflow.types.ckpt import Checkpoint
from visflow.types.ckpt import __vf_ckpt_version__


class VFModel(nn.Module, abc.ABC):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with predictions.
        """

    @property
    def last_conv(self) -> nn.Module:
        """Get the last convolutional layer of the model.

        Returns:
            The last convolutional layer module.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `last_conv` method.")

    def load(self, ckpt: Checkpoint, *, strict: bool = True) -> None:
        """Load model state from checkpoint dictionary.

        Args:
            ckpt: Checkpoint dictionary containing model state and metadata.
            strict: Whether to strictly enforce state dict key matching.
        """
        classes = ckpt.classes
        self.num_classes = len(classes) or self.num_classes
        self.load_state_dict(ckpt.model_state_dict, strict=strict)

    def loads(
        self,
        ckpt_path: str | os.PathLike[str],
        *,
        strict: bool = True,
        map_location: str | torch.device | None = None,
    ) -> None:
        """Load model state from checkpoint file.

        Args:
            ckpt_path: Path to the checkpoint file.
            strict: Whether to strictly enforce state dict key matching.
            map_location: Device location for loading tensors.
        """
        path = pathlib.Path(ckpt_path)
        ckpt = torch.load(path, map_location=map_location or "cpu")
        if not isinstance(ckpt, Checkpoint):
            raise ValueError(f"Invalid checkpoint format in file: {ckpt_path}")
        if not ckpt.__vf_ckpt_version__ == __vf_ckpt_version__:
            raise ValueError(
                f"Checkpoint version mismatch: expected {__vf_ckpt_version__}, "
                f"got {ckpt.__vf_ckpt_version__} in file: {ckpt_path}"
            )
        self.load(ckpt, strict=strict)


REGISTRY: dict[str, type[VFModel]] = {}


_C = t.TypeVar("_C", bound=VFModel)


def export_vf_model(*, name: str) -> t.Callable[[type[_C]], type[_C]]:
    """Decorator to register a vf model class in the global registry.

    Args:
        name: Name to register the model under.

    Returns:
        Decorator function that registers the model class.

    Raises:
        TypeError: If the decorated class doesn't inherit from VFModel.
    """

    def decorator(cls: type[_C]) -> type[_C]:
        global REGISTRY
        if not issubclass(cls, VFModel):
            raise TypeError(f"Cannot register {cls.__name__}: not a subclass of VFModel")
        REGISTRY[name] = cls
        return cls

    return decorator


# Dynamically import all modules in the current package
_package_dir = pathlib.Path(__file__).parent
for _, module_name, ispkg in pkgutil.iter_modules([str(_package_dir)]):
    if not ispkg and module_name not in {"__init__", "base"}:
        importlib.import_module(f"{__name__}.{module_name}")
