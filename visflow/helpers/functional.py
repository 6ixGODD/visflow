from __future__ import annotations

import collections as coll
import functools as ft
import os
import platform
import typing as t

import numpy as np
import psutil
import torch
import torch.nn as nn
import torchvision

from visflow.context import EnvironmentInfo, ModelSummary
from visflow.types import CriterionFunc


class MixUpLoss(nn.Module):
    def __init__(self, criterion: CriterionFunc):
        super().__init__()
        self.criterion = criterion

    def forward(
        self,
        pred: torch.Tensor,
        target_a: torch.Tensor,
        target_b: t.Optional[torch.Tensor] = None,
        lam: float | None = None
    ) -> torch.Tensor:
        if target_b is not None and lam is not None:
            return (lam *
                    self.criterion(pred, target_a) + (1 - lam) *
                    self.criterion(pred, target_b))
        else:
            return self.criterion(pred, target_a)


def summary(
    model: nn.Module,
    input_size: t.Tuple[int, ...] | t.List[t.Tuple[int, ...]],
    batch_size: int = -1,
    device: t.Literal['cuda', 'cpu'] = 'cuda'
) -> ModelSummary:
    def register_hook(module: nn.Module) -> None:
        def hook(
            module: nn.Module,
            input: t.Tuple[torch.Tensor, ...],
            output: torch.Tensor
        ) -> None:
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(layer_summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            layer_summary[m_key] = {}
            layer_summary[m_key]["input_shape"] = list(input[0].size())
            layer_summary[m_key]["input_shape"][0] = batch_size
            layer_summary[m_key]["output_shape"] = list(output.size())
            layer_summary[m_key]["output_shape"][0] = batch_size

            params = torch.tensor(0)
            trainable = False
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(
                    torch.LongTensor(list(module.weight.size()))  # type: ignore
                )
                trainable = bool(module.weight.requires_grad)
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(
                    torch.LongTensor(list(module.bias.size()))  # type: ignore
                )

            layer_summary[m_key]["nb_params"] = int(params)
            layer_summary[m_key]["trainable"] = trainable

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

    # create properties
    layer_summary = coll.OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    # Calculate statistics
    total_params = 0
    total_output = 0
    trainable_params = 0

    for layer in layer_summary:
        total_params += layer_summary[layer]["nb_params"]
        if isinstance(layer_summary[layer]["output_shape"][0], list):
            # Handle multiple outputs
            for output_shape in layer_summary[layer]["output_shape"]:
                total_output += np.prod(output_shape)
        else:
            total_output += np.prod(layer_summary[layer]["output_shape"])

        if layer_summary[layer]["trainable"]:
            trainable_params += layer_summary[layer]["nb_params"]

    # Calculate memory sizes (assume 4 bytes/number - float32)
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(
        2. * total_output * 4. / (1024 ** 2.)
    )  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    return ModelSummary(
        layers=layer_summary,
        total_params=total_params,
        trainable_params=trainable_params,
        non_trainable_params=total_params - trainable_params,
        input_size_mb=round(total_input_size, 2),
        forward_backward_pass_size_mb=round(total_output_size, 2),
        params_size_mb=round(total_params_size, 2),
        estimated_total_size_mb=round(total_size, 2)
    )


@ft.lru_cache(maxsize=1)
def env_info() -> EnvironmentInfo:
    return EnvironmentInfo(
        pid=os.getpid(),
        os=platform.system(),
        cpu_cores=psutil.cpu_count(logical=True),
        cpu_usage=psutil.cpu_percent(interval=1),
        cpu_architecture=platform.machine(),
        memory_gb=round(psutil.virtual_memory().total / (1024 ** 3), 2),
        memory_usage_gb=round(
            psutil.virtual_memory().used / (1024 ** 3), 2
        ),
        gpu_available=torch.cuda.is_available(),
        gpu_model=(torch.cuda.get_device_name(0)
                   if torch.cuda.is_available()
                   else "N/A"),
        gpu_memory_gb=round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
        ) if torch.cuda.is_available() else 0,
        gpu_memory_usage_gb=round(
            torch.cuda.memory_allocated(0) / (1024 ** 3), 2
        ) if torch.cuda.is_available() else 0,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        torchvision_version=torchvision.__version__,
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
    )
