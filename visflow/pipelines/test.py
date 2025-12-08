from __future__ import annotations

import pathlib as p
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from visflow.config import TestConfig
from visflow.config import TrainConfig
from visflow.context import CheckpointTestLog
from visflow.context import Context
from visflow.context import Metrics
from visflow.context import TestEndLog
from visflow.data import Datamodule
from visflow.helper.display import Display
from visflow.helper.functional import env_info
from visflow.helper.metrics import compute_metric
from visflow.helper.plotting import plot_combined_roc_curves
from visflow.logger import Logger
from visflow.logger.types import LoggingTarget
from visflow.models import BaseClassifier
from visflow.pipelines import BasePipeline
from visflow.types import Checkpoint
from visflow.utils import gen_id
from visflow.utils import incr_path
from visflow.utils import seed
from visflow.utils import spinner


class TestPipeline(BasePipeline):
    __slots__ = (
        *BasePipeline.__slots__,
        "config",
        "logger",
        "device",
        "datamodule",
        "checkpoint_results",
        "start_time",
    )

    def __init__(self, config: TestConfig):
        self._completed = False
        self.config = config
        self.logger = Logger(config.logging)
        self.device = torch.device(config.device)

        # Will be initialized from first checkpoint
        self.checkpoint_results: list[CheckpointTestLog] = []
        self.start_time = 0.0

    def __call__(self) -> None:
        self.start_time = time.time()

        # Setup experiment -------------------------------------------------------------------------
        spinner.start("Setting up experiment...")
        seed(self.config.seed)
        exp_id = gen_id(pref="test", without_hyphen=True)

        output_dir = p.Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exp_dir = incr_path(output_dir, self.config.experiment_name)

        context = Context(
            experiment_id=exp_id,
            experiment_name="test_comparison",
            timestamp=time.strftime("%Y%m%dT%H%M%S", time.localtime()),
        )
        Display(context)
        self.logger.add_target(LoggingTarget(logname=exp_dir / ".log.json", loglevel="info"))
        logger = self.logger.with_context(TAG="test", **context)
        env_info()
        spinner.succeed("Experiment setup completed.")

        # Store results for combined plotting
        checkpoint_outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        # Test each checkpoint
        for i, checkpoint_path in enumerate(self.config.checkpoint_files):
            spinner.start(
                f"Testing checkpoint {i + 1}/{len(self.config.checkpoint_files)}: "
                f"{p.Path(checkpoint_path).name}"
            )
            ckpt = self.load(checkpoint_path)

            checkpoint_result = self.test_ckpt(
                checkpoint_path=checkpoint_path,
                test_loader=Datamodule(TrainConfig(**ckpt["config"])),
                logger=logger,
                exp_dir=exp_dir,
            )

            self.checkpoint_results.append(checkpoint_result)

            # Store outputs for combined plotting (assuming this info is
            # stored in checkpoint_result)
            # You'll need to modify _test_checkpoint to return this
            checkpoint_name = checkpoint_result["checkpoint_name"]
            # This would need to be returned from _test_checkpoint:
            # checkpoint_outputs[checkpoint_name] = (test_targets, test_outputs)

            spinner.succeed(f"Checkpoint {checkpoint_name} tested successfully.")

        # Generate combined visualizations
        spinner.start("Generating comparison visualizations...")
        self.plots(exp_dir, checkpoint_outputs)

        # Save comprehensive results
        self.save(exp_dir)

        # Log final results
        best_checkpoint = max(self.checkpoint_results, key=lambda x: x["test_metrics"]["accuracy"])
        total_time = time.time() - self.start_time

        end_log = TestEndLog(
            total_checkpoints=len(self.config.checkpoint_files),
            total_time_sec=total_time,
            checkpoint_results=self.checkpoint_results,
            best_checkpoint=best_checkpoint,
            test_dataset_info=self.datamodule.info if self.datamodule else {},
        )

        logger.info("Test pipeline completed", **end_log)
        spinner.succeed("Test pipeline completed successfully.")
        self._completed = True

    def load(self, ckpt_path: str) -> Checkpoint:
        if not p.Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint: Checkpoint = torch.load(ckpt_path, map_location=self.device)
        return checkpoint

    def eval(
        self, model: BaseClassifier, dataloader: DataLoader
    ) -> tuple[Metrics, torch.Tensor, torch.Tensor]:
        """Evaluate model on dataloader."""
        model.eval()
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(inputs)

                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())

        outputs_tensor = torch.cat(all_outputs, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)

        # Compute metrics
        metrics = compute_metric(outputs_tensor, targets_tensor)

        return metrics, outputs_tensor, targets_tensor

    def plots(
        self, exp_dir: p.Path, checkpoint_outputs: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """Generate comparison plots for all checkpoints."""
        if not checkpoint_outputs:
            return

        # Get number of classes from first checkpoint
        first_outputs = next(iter(checkpoint_outputs.values()))[1]
        num_classes = first_outputs.shape[1]

        # Get class names from first checkpoint (you might need to store this)
        class_names = None  # You'll need to get this from somewhere

        # Generate combined ROC curves
        roc_path = exp_dir / "combined_roc_curves.png"
        auc_scores = plot_combined_roc_curves(
            checkpoint_results=checkpoint_outputs,
            num_classes=num_classes,
            class_names=class_names,
            save_path=roc_path,
            show=False,
        )

        # Save AUC scores to file
        auc_df = pd.DataFrame(auc_scores).T
        auc_df.to_csv(exp_dir / "auc_scores_comparison.csv")

    def save(self, exp_dir: p.Path) -> None:
        """Save comprehensive comparison results."""
        # Create results DataFrame
        results_data = []
        for result in self.checkpoint_results:
            row = {
                "checkpoint_name": result["checkpoint_name"],
                "epoch": result["epoch"],
                "accuracy": result["test_metrics"]["accuracy"],
                "precision": result["test_metrics"]["precision"],
                "recall": result["test_metrics"]["recall"],
                "f1_score": result["test_metrics"]["f1_score"],
                "inference_time_sec": result["inference_time_sec"],
                "model_params": result["model_params"],
            }
            results_data.append(row)

        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values("accuracy", ascending=False)

        # Save to CSV
        results_df.to_csv(exp_dir / "checkpoint_comparison.csv", index=False)

        # Save detailed results as JSON
        import json

        with (exp_dir / "detailed_results.json").open("w") as f:
            json.dump(self.checkpoint_results, f, indent=2, default=str)
