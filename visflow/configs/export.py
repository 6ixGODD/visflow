from __future__ import annotations

import pydantic as pydt


class ExportConfig(pydt.BaseModel):
    export_onnx: bool = pydt.Field(
        default=False,
        description="Whether to export the trained model to ONNX format."
    )

    onnx_opset: int = pydt.Field(
        default=11,
        ge=7,
        le=17,
        description="ONNX opset version for model export."
    )

    onnx_dynamic_axes: bool = pydt.Field(
        default=True,
        description="Whether to enable dynamic batch size in ONNX export."
    )

    export_torchscript: bool = pydt.Field(
        default=False,
        description="Whether to export the trained model to TorchScript format."
    )
