from __future__ import annotations

import typing as t

import pydantic as pydt


class TrainingConfig(pydt.BaseModel):
    batch_size: int = pydt.Field(
        default=32,
        ge=1,
        le=512,
        description="Number of samples processed in each training batch."
    )

    epochs: int = pydt.Field(
        default=10,
        ge=1,
        description="Maximum number of training epochs."
    )

    learning_rate: float = pydt.Field(
        default=1e-3,
        gt=0.0,
        le=1.0,
        description="Initial learning rate for optimization."
    )

    momentum: float = pydt.Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Momentum factor for SGD optimizer."
    )

    weight_decay: float = pydt.Field(
        default=1e-4,
        ge=0.0,
        description="L2 regularization strength to prevent overfitting."
    )

    optimizer: t.Literal['sgd', 'adam', 'adamw'] = pydt.Field(
        default='adam',
        description="Optimization algorithm."
    )

    lr_scheduler: t.Literal[
        'step', 'multistep', 'cosine', 'plateau', 'none'
    ] = pydt.Field(
        default='step',
        description="Learning rate scheduling strategy."
    )

    early_stopping: bool = pydt.Field(
        default=True,
        description="Whether to enable early stopping based on validation "
                    "performance."
    )

    early_stopping_patience: int = pydt.Field(
        default=5,
        ge=1,
        description="Number of epochs with no improvement after which "
                    "training will be stopped."
    )

    label_smoothing: float = pydt.Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Label smoothing factor to prevent overconfident "
                    "predictions."
    )
