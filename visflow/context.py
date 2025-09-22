from __future__ import annotations

import typing as t

import typing_extensions as te


class Context(te.TypedDict):
    experiment_id: str
    """Unique identifier for the experiment."""

    experiment_name: str
    """Human-readable name for the experiment."""

    timestamp: str
    """ISO 8601 formatted timestamp of the event."""


class EnvironmentInfo(te.TypedDict, total=False):
    pid: te.Required[int]
    """Process ID of the running experiment."""

    os: te.Required[str]
    """Operating system name and version."""

    cpu_cores: te.Required[int]
    """Number of CPU cores available."""

    cpu_usage: float
    """Current CPU usage percentage."""

    cpu_architecture: te.Required[str]
    """CPU architecture (e.g., x86_64, ARM)."""

    memory_gb: te.Required[float]
    """Total memory in gigabytes."""

    memory_usage_gb: float
    """Current memory usage in gigabytes."""

    gpu_available: te.Required[bool]
    """Indicates if a GPU is available."""

    gpu_model: str
    """Model of the GPU if available."""

    gpu_memory_gb: float
    """Total GPU memory in gigabytes if available."""

    gpu_memory_usage_gb: float
    """Current GPU memory usage in gigabytes if available."""

    python_version: te.Required[str]
    """Version of Python being used."""

    torch_version: te.Required[str]
    """Version of PyTorch being used."""

    torchvision_version: te.Required[str]
    """Version of TorchVision being used."""

    cuda_version: str
    """Version of CUDA if available."""

    cudnn_version: str
    """Version of cuDNN if available."""


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


class Metrics(te.TypedDict, total=False):
    loss: float | None
    """Current loss value."""

    accuracy: float
    """Current accuracy value."""

    precision: float
    """Current precision value."""

    recall: float
    """Current recall value."""

    f1_score: float
    """Current F1 score value."""

    auc_roc: float
    """Current AUC-ROC value."""

    confusion_matrix: t.List[t.List[int]]
    """Current confusion matrix."""

    sensitivity: float
    """Current sensitivity value."""

    specificity: float
    """Current specificity value."""

    extras: t.Dict[str, float]
    """Any additional metrics."""


class BatchLog(te.TypedDict, total=False):
    epoch: te.Required[int]
    """Current epoch number."""

    total_epochs: te.Required[int]
    """Total number of epochs."""

    batch: te.Required[int]
    """Current batch number within the epoch."""

    total_batches: te.Required[int]
    """Total number of batches in the epoch."""

    metrics: te.Required[Metrics]
    """Metrics for the current batch.
    
    !NOTE: These metrics are typically noisy and may not reflect overall 
    performance."""

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


class EpochLog(te.TypedDict):
    epoch: te.Required[int]
    """Current epoch number."""

    total_epochs: te.Required[int]
    """Total number of epochs."""

    train_metrics: te.Required[Metrics]
    """Average training metrics for the epoch."""

    val_metrics: te.Required[Metrics]
    """Average validation metrics for the epoch."""

    best_val_metrics: te.Required[Metrics | None]
    """Best validation metrics achieved so far."""

    best_epoch: int
    """Epoch number where the best validation metrics were achieved."""

    epoch_time_sec: float
    """Time taken to complete the epoch in seconds."""

    initial_lr: float
    """Initial learning rate for the epoch."""

    final_lr: float
    """Final learning rate for the epoch."""


class LayerInfo(te.TypedDict):
    input_shape: t.List[int]
    """Shape of the input tensor."""

    output_shape: t.List[int] | t.List[t.List[int]]
    """Shape of the output tensor."""

    nb_params: int
    """Number of parameters in the layer."""

    trainable: bool
    """Indicates if the layer is trainable."""


class ModelSummary(te.TypedDict):
    layers: t.Dict[str, LayerInfo]
    """Information about each layer in the model."""

    total_params: int
    """Total number of parameters in the model."""

    trainable_params: int
    """Number of trainable parameters in the model."""

    non_trainable_params: int
    """Number of non-trainable parameters in the model."""

    input_size_mb: float
    """Size of the input tensor in megabytes."""

    forward_backward_pass_size_mb: float
    """Size of the forward and backward pass in megabytes."""

    params_size_mb: float
    """Size of the parameters in megabytes."""

    estimated_total_size_mb: float
    """Estimated total size of the model in megabytes."""


class ExperimentStartLog(te.TypedDict, total=False):
    env: te.Required[EnvironmentInfo]
    """Information about the hardware used."""

    dataset: te.Required[DatasetInfo]
    """Information about the dataset."""

    config: te.Required[t.Dict[str, t.Any]]
    """Configuration parameters for the experiment."""

    model_summary: te.Required[ModelSummary]
    """Summary of the model architecture."""


class ExperimentEndLog(te.TypedDict, total=False):
    total_epochs: te.Required[int]
    """Total number of epochs completed."""

    total_time_sec: te.Required[float]
    """Total time taken for the experiment in seconds."""

    final_metrics: te.Required[Metrics]
    """Final metrics achieved at the end of the experiment."""

    best_metrics: te.Required[Metrics]
    """Best metrics achieved during the experiment."""

    test_metrics: Metrics
    """Metrics evaluated on the test set."""

    best_epoch: te.Required[int]
    """Epoch number at which the best metrics were achieved."""


class CheckpointTestLog(te.TypedDict):
    """Log for individual checkpoint testing results."""

    checkpoint_path: te.Required[str]
    """Path to the checkpoint file."""

    checkpoint_name: te.Required[str]
    """Name of the checkpoint (derived from filename)."""

    epoch: te.Required[int]
    """Epoch number from the checkpoint."""

    test_metrics: te.Required[Metrics]
    """Test metrics for this checkpoint."""

    inference_time_sec: float
    """Time taken for inference in seconds."""

    model_params: int
    """Number of model parameters."""


class TestEndLog(te.TypedDict):
    """Log for test pipeline completion."""

    total_checkpoints: te.Required[int]
    """Total number of checkpoints tested."""

    total_time_sec: te.Required[float]
    """Total time taken for all tests."""

    checkpoint_results: te.Required[t.List[CheckpointTestLog]]
    """Results for all checkpoints."""

    best_checkpoint: te.Required[CheckpointTestLog]
    """Best performing checkpoint."""

    test_dataset_info: t.Dict[str, t.Any]
    """Information about the test dataset."""
