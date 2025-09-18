from __future__ import annotations

import datetime
import os
import time
import typing as t

import pyfiglet
import rich.align as ralign
import rich.box as rbox
import rich.console as rconsole
import rich.panel as rpanel
import rich.table as rtable
import rich.text as rtext

from visflow import __project__
from visflow.context import (
    Context,
    DatasetInfo,
    EnvironmentInfo,
    EpochLog,
    ExperimentEndLog,
    ExperimentStartLog,
    LayerInfo,
    ModelSummary,
)
from visflow.utils import flatten_dict


def make_model_summary_table(model_summary: ModelSummary) -> rtable.Table:
    table = rtable.Table(box=rbox.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Total Parameters", f"{model_summary['total_params']:,}")
    table.add_row("Trainable Parameters", f"{model_summary['trainable_params']:,}")
    table.add_row(
        "Non-trainable Parameters", f"{model_summary['non_trainable_params']:,}"
    )
    table.add_row("Input Size (MB)", f"{model_summary['input_size_mb']:.2f}")
    table.add_row(
        "Forward/Backward Pass (MB)",
        f"{model_summary['forward_backward_pass_size_mb']:.2f}",
    )
    table.add_row("Parameters Size (MB)", f"{model_summary['params_size_mb']:.2f}")
    table.add_row(
        "Estimated Total Size (MB)", f"{model_summary['estimated_total_size_mb']:.2f}"
    )

    return table


def make_final_metrics_table(end_log: ExperimentEndLog) -> rtable.Table:
    table = rtable.Table(box=rbox.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Final", style="white")
    table.add_column("Best", style="green")

    final_metrics = end_log["final_metrics"]
    best_metrics = end_log["best_metrics"]

    for metric_name in [
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auc_roc",
    ]:
        final_val = final_metrics.get(metric_name)  # type: ignore
        best_val = best_metrics.get(metric_name)  # type: ignore

        if final_val is not None or best_val is not None:
            final_str = f"{final_val:.4f}" if final_val is not None else "N/A"
            best_str = f"{best_val:.4f}" if best_val is not None else "N/A"
            table.add_row(metric_name.replace("_", " ").title(), final_str, best_str)

    # Add experiment summary
    table.add_row("", "", "")  # Separator
    table.add_row("Total Epochs", str(end_log["total_epochs"]), "")
    table.add_row("Best Epoch", "", str(end_log["best_epoch"]))
    table.add_row("Total Time (sec)", f"{end_log['total_time_sec']:.2f}", "")

    return table


def make_ascii_art() -> str:
    try:
        ascii_art = pyfiglet.figlet_format(__project__, font="slant")
        return str(ascii_art)
    except:
        return __project__


def make_env_table(env_info: EnvironmentInfo) -> rtable.Table:
    """Create environment information table."""
    table = rtable.Table(box=rbox.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Process ID", str(env_info["pid"]))
    table.add_row("Operating System", env_info["os"])
    table.add_row("CPU Architecture", env_info["cpu_architecture"])
    table.add_row("CPU Cores", str(env_info["cpu_cores"]))

    if "cpu_usage" in env_info:
        table.add_row("CPU Usage (%)", f"{env_info['cpu_usage']:.1f}")

    table.add_row("Memory (GB)", f"{env_info['memory_gb']:.1f}")

    if "memory_usage_gb" in env_info:
        table.add_row("Memory Usage (GB)", f"{env_info['memory_usage_gb']:.1f}")

    table.add_row("GPU Available", "✓" if env_info["gpu_available"] else "✗")

    if env_info["gpu_available"] and "gpu_model" in env_info:
        table.add_row("GPU Model", env_info["gpu_model"])
        if "gpu_memory_gb" in env_info:
            table.add_row("GPU Memory (GB)", f"{env_info['gpu_memory_gb']:.1f}")
        if "gpu_memory_usage_gb" in env_info:
            table.add_row(
                "GPU Memory Usage (GB)", f"{env_info['gpu_memory_usage_gb']:.1f}"
            )

    table.add_row("Python Version", env_info["python_version"])
    table.add_row("PyTorch Version", env_info["torch_version"])
    table.add_row("TorchVision Version", env_info["torchvision_version"])

    if "cuda_version" in env_info:
        table.add_row("CUDA Version", env_info["cuda_version"])

    if "cudnn_version" in env_info:
        table.add_row("cuDNN Version", env_info["cudnn_version"])

    return table


def make_dataset_table(dataset_info: DatasetInfo) -> rtable.Table:
    table = rtable.Table(box=rbox.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Number of Classes", str(dataset_info["num_classes"]))
    table.add_row("Training Samples", f"{dataset_info['train_size']:,}")
    table.add_row("Validation Samples", f"{dataset_info['val_size']:,}")
    table.add_row("Test Samples", f"{dataset_info['test_size']:,}")

    # Show first few class names if available
    if dataset_info["classes"]:
        class_preview = ", ".join(dataset_info["classes"][:5])
        if len(dataset_info["classes"]) > 5:
            class_preview += f" ... (+{len(dataset_info['classes']) - 5} " f"more)"
        table.add_row("Class Names", class_preview)

    return table


def make_config_table(config: t.Dict[str, t.Any]) -> rtable.Table:
    """Create configuration table."""
    table = rtable.Table(box=rbox.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    # Flatten the config dict to handle nested configurations
    flat_config = flatten_dict(config)

    for key, value in flat_config.items():
        # Format the value appropriately
        if isinstance(value, float):
            if 0.001 > value > 0:
                value_str = f"{value:.6f}"
            else:
                value_str = f"{value:.4f}"
        elif isinstance(value, (list, tuple)):
            if len(str(value)) > 50:
                value_str = f"{str(value)[:47]}..."
            else:
                value_str = str(value)
        else:
            value_str = str(value)

        table.add_row(key, value_str)

    return table


def make_layers_table(layers: t.Dict[str, LayerInfo]) -> rtable.Table:
    table = rtable.Table(box=rbox.SIMPLE, show_header=True, padding=(0, 1))
    table.add_column("Layer (type)", style="white", no_wrap=False, min_width=25)
    table.add_column("Output Shape", style="white", justify="right", min_width=25)
    table.add_column("Param #", style="white", justify="right", min_width=15)

    for layer_name, layer_info in layers.items():
        # Format output shape to match the original style
        output_shape = layer_info["output_shape"]
        if isinstance(output_shape, list) and len(output_shape) > 0:
            if isinstance(output_shape[0], list):
                # Multiple output shapes
                output_shape_str = str(output_shape)
            else:
                # Single output shape - format as list
                output_shape_str = str(output_shape)
        else:
            output_shape_str = str(output_shape)

        param_count = f"{layer_info['nb_params']:,}"

        table.add_row(layer_name, output_shape_str, param_count)

    return table


def make_experiment_info_table(
    exp_id: str, exp_name: str, timestamp: str
) -> rtable.Table:
    table = rtable.Table(box=rbox.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Experiment ID", exp_id)
    table.add_row("Experiment Name", exp_name)
    table.add_row("Start Time", timestamp)

    current_time = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    table.add_row("Current Time (UTC)", current_time)

    return table


class Display:
    def __init__(self, context: Context):
        self.console = rconsole.Console()
        self.context = context
        self.start_time = 0.0

    def display_start(self, start_log: ExperimentStartLog) -> None:
        self.clear()
        self.start_time = time.time()
        self.console.clear()

        # ASCII Art
        ascii_art = make_ascii_art()
        ascii_panel = rpanel.Panel(
            ralign.Align.center(rtext.Text(ascii_art, style="bold blue")),
            box=rbox.DOUBLE,
            padding=(1, 2),
        )
        self.console.print(ascii_panel)
        self.console.print()

        # Experiment Information
        exp_info = make_experiment_info_table(
            self.context.get("experiment_id", "N/A"),
            self.context.get("experiment_name", "N/A"),
            self.context.get("timestamp", "N/A"),
        )
        exp_panel = rpanel.Panel(
            exp_info,
            title="[bold cyan]Experiment Information[/bold cyan]",
            box=rbox.ROUNDED,
            padding=(1, 2),
        )
        self.console.print(exp_panel)
        self.console.print()

        # Environment Information
        env_table = make_env_table(start_log["env"])
        env_panel = rpanel.Panel(
            env_table,
            title="[bold cyan]Environment Information[/bold cyan]",
            box=rbox.ROUNDED,
            padding=(1, 2),
        )
        self.console.print(env_panel)
        self.console.print()

        # Dataset Information
        dataset_table = make_dataset_table(start_log["dataset"])
        dataset_panel = rpanel.Panel(
            dataset_table,
            title="[bold cyan]Dataset Information[/bold cyan]",
            box=rbox.ROUNDED,
            padding=(1, 2),
        )
        self.console.print(dataset_panel)
        self.console.print()

        # Configuration Information
        config_table = make_config_table(start_log["config"])
        config_panel = rpanel.Panel(
            config_table,
            title="[bold cyan]Configuration[/bold cyan]",
            box=rbox.ROUNDED,
            padding=(1, 2),
        )
        self.console.print(config_panel)
        self.console.print()

        # Model Architecture (if layers are provided)
        if start_log["model_summary"]["layers"]:
            layers_table = make_layers_table(start_log["model_summary"]["layers"])
            layers_panel = rpanel.Panel(
                layers_table,
                title="[bold cyan]Model Architecture[/bold cyan]",
                box=rbox.ROUNDED,
                padding=(1, 2),
            )
            self.console.print(layers_panel)
            self.console.print()

        # Model Summary
        model_table = make_model_summary_table(start_log["model_summary"])
        model_panel = rpanel.Panel(
            model_table,
            title="[bold cyan]Model Summary[/bold cyan]",
            box=rbox.ROUNDED,
            padding=(1, 2),
        )
        self.console.print(model_panel)
        self.console.print()

        # Separator
        separator = "─" * 80
        self.console.print(rtext.Text(separator, style="dim"), justify="center")
        self.console.print(
            rtext.Text("🚀 Experiment Started", style="bold green"), justify="center"
        )
        self.console.print(rtext.Text(separator, style="dim"), justify="center")
        self.console.print()

    def display_metrics(self, epoch_log: EpochLog) -> None:
        """Display metrics for the current epoch."""
        current_epoch = epoch_log["epoch"]
        total_epochs = epoch_log["total_epochs"]

        # Create progress indicator
        progress_bar = "█" * int((current_epoch / total_epochs) * 40)
        remaining_bar = "░" * (40 - int((current_epoch / total_epochs) * 40))
        progress_str = (
            f"[{progress_bar}{remaining_bar}] {current_epoch}/" f"{total_epochs}"
        )

        # Epoch header
        epoch_header = f"Epoch {current_epoch}/{total_epochs}"
        if "epoch_time_sec" in epoch_log:
            epoch_header += f" | Time: {epoch_log['epoch_time_sec']:.2f}s"

        self.console.print()
        self.console.print(rtext.Text(epoch_header, style="bold green"))
        self.console.print(rtext.Text(progress_str, style="white"))

        # Current epoch metrics table
        metrics_table = rtable.Table(box=rbox.SIMPLE, show_header=True, padding=(0, 1))
        metrics_table.add_column("Metric", style="cyan", no_wrap=True)
        metrics_table.add_column("Train", style="yellow", justify="right")
        metrics_table.add_column("Val", style="white", justify="right")
        metrics_table.add_column("Best Val", style="green", justify="right")

        train_metrics = epoch_log["train_metrics"]
        val_metrics = epoch_log["val_metrics"]
        best_val_metrics = epoch_log["best_val_metrics"]

        # Display main metrics
        for metric_name in [
            "loss",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "auc_roc",
        ]:
            train_val = train_metrics.get(metric_name)  # type: ignore
            val_val = val_metrics.get(metric_name)  # type: ignore
            best_val = best_val_metrics.get(metric_name)  # type: ignore

            if train_val is not None or val_val is not None:
                train_str = f"{train_val:.4f}" if train_val is not None else "N/A"
                val_str = f"{val_val:.4f}" if val_val is not None else "N/A"
                best_str = f"{best_val:.4f}" if best_val is not None else "N/A"

                # Add visual indicator for improvement
                indicator = ""
                if best_val is not None and val_val is not None and val_val == best_val:
                    if metric_name == "loss":
                        indicator = " ⬇️"  # Lower is better for loss
                    else:
                        indicator = " ⬆️"  # Higher is better for other metrics

                metrics_table.add_row(
                    metric_name.replace("_", " ").title(),
                    train_str,
                    val_str + indicator,
                    best_str,
                )

        # Display extra metrics if available
        train_extras = (
            train_metrics.get("extras", {}) if "extras" in train_metrics else {}
        )
        val_extras = val_metrics.get("extras", {}) if "extras" in val_metrics else {}
        best_extras = (
            best_val_metrics.get("extras", {})
            if best_val_metrics and "extras" in best_val_metrics
            else {}
        )

        all_extra_keys = (
            set(train_extras.keys()) | set(val_extras.keys()) | set(best_extras.keys())
        )

        if all_extra_keys:
            metrics_table.add_row("", "", "", "")  # Separator
            for extra_metric in sorted(all_extra_keys):
                train_extra = train_extras.get(extra_metric)
                val_extra = val_extras.get(extra_metric)
                best_extra = best_extras.get(extra_metric)

                train_str = f"{train_extra:.4f}" if train_extra is not None else "N/A"
                val_str = f"{val_extra:.4f}" if val_extra is not None else "N/A"
                best_str = f"{best_extra:.4f}" if best_extra is not None else "N/A"

                metrics_table.add_row(
                    extra_metric.replace("_", " ").title(), train_str, val_str, best_str
                )

        self.console.print(metrics_table)

        # Learning rate and best epoch information
        info_table = rtable.Table(box=rbox.SIMPLE, show_header=False, padding=(0, 1))
        info_table.add_column("Key", style="cyan", no_wrap=True)
        info_table.add_column("Value", style="white")

        if "initial_lr" in epoch_log:
            info_table.add_row("Initial LR", f"{epoch_log['initial_lr']:.6f}")
        if "final_lr" in epoch_log:
            info_table.add_row("Final LR", f"{epoch_log['final_lr']:.6f}")

        if "best_epoch" in epoch_log:
            info_table.add_row("Best Epoch", str(epoch_log["best_epoch"]))

        if info_table.row_count > 0:
            self.console.print(info_table)

        # Add a subtle separator between epochs
        separator = "─" * 60
        self.console.print(rtext.Text(separator, style="dim"))

    def display_end(self, end_log: ExperimentEndLog) -> None:
        end_time = time.time()
        self.console.print()

        # Separator
        separator = "─" * 80
        self.console.print(rtext.Text(separator, style="dim"), justify="center")
        self.console.print(
            rtext.Text("🎯 Experiment Completed", style="bold green"), justify="center"
        )
        self.console.print(rtext.Text(separator, style="dim"), justify="center")
        self.console.print()

        # Final Results
        results_table = make_final_metrics_table(end_log)
        results_panel = rpanel.Panel(
            results_table,
            title="[bold cyan]Experiment Results[/bold cyan]",
            box=rbox.ROUNDED,
            padding=(1, 2),
        )
        self.console.print(results_panel)
        self.console.print()

        # Execution time
        execution_time = end_time - self.start_time
        time_table = rtable.Table(box=rbox.SIMPLE, show_header=False, padding=(0, 1))
        time_table.add_column("Key", style="cyan", no_wrap=True)
        time_table.add_column("Value", style="white")
        time_table.add_row("Total Execution Time", f"{execution_time:.2f} seconds")

        time_panel = rpanel.Panel(
            time_table,
            title="[bold cyan]Execution Summary[/bold cyan]",
            box=rbox.ROUNDED,
            padding=(1, 2),
        )
        self.console.print(time_panel)

        # Final ASCII art (smaller)
        try:
            final_art = pyfiglet.figlet_format("Complete!", font="small")
            final_panel = rpanel.Panel(
                ralign.Align.center(rtext.Text(final_art, style="bold green")),
                box=rbox.SIMPLE,
                padding=(0, 1),
            )
            self.console.print(final_panel)
        except:
            self.console.print(
                rtext.Text("✨ Experiment Complete! ✨", style="bold green"),
                justify="center",
            )

    @staticmethod
    def clear() -> None:
        os.system("cls" if os.name == "nt" else "clear")
