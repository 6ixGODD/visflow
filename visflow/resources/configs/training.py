from __future__ import annotations

import typing as t

import pydantic as pydt


class TrainingConfig(pydt.BaseModel):
    device: t.Literal['cpu', 'cuda'] = pydt.Field(
        default='cuda',
        description="Device to use for training. Options are 'cpu' or 'cuda'."
    )

    shuffle: bool = pydt.Field(
        default=True,
        description="Whether to shuffle the dataset at the beginning of "
                    "each epoch."
    )

    batch_size: int = pydt.Field(
        default=32,
        ge=1,
        le=512,
        description="Number of samples processed in each training batch."
    )

    weighted_sampling: bool = pydt.Field(
        default=False,
        description="Whether to use weighted sampling to handle class "
                    "imbalance."
    )

    drop_last: bool = pydt.Field(
        default=False,
        description="Whether to drop the last incomplete batch if the "
                    "dataset size is not divisible by the batch size."
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
