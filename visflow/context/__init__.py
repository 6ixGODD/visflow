from __future__ import annotations

import typing as t

import typing_extensions as te


class BaseContext(te.TypedDict):
    experiment_id: str
    """Unique identifier for the experiment."""

    experiment_name: str
    """Human-readable name for the experiment."""

    timestamp: str
    """ISO 8601 formatted timestamp of the event."""


class HardwareInfo(te.TypedDict, total=False):
    cpu_cores: te.Required[int]
    """Number of CPU cores available."""

    memory_gb: te.Required[float]
    """Total memory in gigabytes."""

    gpu_available: te.Required[bool]
    """Indicates if a GPU is available."""

    gpu_model: str
    """Model of the GPU if available."""

    gpu_memory_gb: float
    """Total GPU memory in gigabytes if available."""

    python_version: te.Required[str]
    """Version of Python being used."""

    torch_version: te.Required[str]
    """Version of PyTorch being used."""

    cuda_version: str
    """Version of CUDA if available."""


class DatasetInfo(te.TypedDict):
    num_classes: int
    """Number of classes in the dataset."""

    train_size: int
    """Number of training samples."""

    val_size: int
    """Number of validation samples."""

    test_size: int
    """Number of test samples."""

    classes: t.List[str]
    """List of class names."""


class Metrics(te.TypedDict):
    loss: float | None
    """Current loss value."""

    accuracy: float | None
    """Current accuracy value."""

    precision: float | None
    """Current precision value."""

    recall: float | None
    """Current recall value."""

    f1_score: float | None
    """Current F1 score value."""

    auc_roc: float | None
    """Current AUC-ROC value."""

    confusion_matrix: t.List[t.List[int]] | None
    """Current confusion matrix."""

    extras: t.Dict[str, float] | None
    """Any additional metrics."""


class BatchLog(BaseContext, total=False):
    epoch: te.Required[int]
    """Current epoch number."""

    batch: te.Required[int]
    """Current batch number within the epoch."""

    total_batches: te.Required[int]
    """Total number of batches in the epoch."""

    metrics: te.Required[Metrics]
    """Metrics for the current batch."""

    learning_rate: float
    """Current learning rate."""

    gradient_norm: float
    """Norm of the gradients."""

    gpu_memory_usage_gb: float
    """GPU memory usage in gigabytes."""

    batch_time_sec: float
    """Time taken to process the batch in seconds."""

    forward_time_sec: float
    """Time taken for the forward pass in seconds."""

    backward_time_sec: float
    """Time taken for the backward pass in seconds."""

    samples_per_sec: float
    """Number of samples processed per second."""


class EpochLog(BaseContext):
    epoch: te.Required[int]
    """Current epoch number."""

    total_epochs: te.Required[int]
    """Total number of epochs."""

    avg_metrics: te.Required[Metrics]
    """Average metrics for the epoch."""

    best_metrics: te.Required[Metrics]
    """Best metrics achieved so far."""

    epoch_time_sec: float
    """Time taken to complete the epoch in seconds."""

    initial_lr: float
    """Initial learning rate for the epoch."""

    final_lr: float
    """Final learning rate for the epoch."""
