from __future__ import annotations

import pathlib as p_
import time
import typing as t

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from visflow.context import CheckpointTestLog, Context, Metrics, TestEndLog
from visflow.data import ImageDatamodule
from visflow.helpers.functional import env_info
from visflow.helpers.metrics import compute_metric
from visflow.helpers.plotting import (
    plot_combined_roc_curves,
    plot_confusion_matrix,
)
from visflow.pipelines import BasePipeline
from visflow.resources.config import TestConfig
from visflow.resources.logger import Logger
from visflow.resources.logger.types import LoggingTarget
from visflow.resources.models import make_model
from visflow.types import Checkpoint
from visflow.utils import gen_id, incr_path, spinner


class TestPipeline(BasePipeline):
    __slots__ = BasePipeline.__slots__ + (
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
        self.checkpoint_results = []  # type: t.List[CheckpointTestLog]
        self.start_time = 0.0

    def __call__(self) -> None:
        self.start_time = time.time()

        # Setup experiment
        spinner.start("Setting up test experiment...")
        exp_id = gen_id(pref="test", without_hyphen=True)

        output_dir = p_.Path("./test_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        exp_dir = incr_path(output_dir, "test_comparison")

        context = Context(
            experiment_id=exp_id,
            experiment_name="test_comparison",
            timestamp=time.strftime("%Y%m%dT%H%M%S", time.localtime()),
        )

        self.logger.add_target(LoggingTarget(logname=exp_dir / ".log.json", loglevel="info"),)
        logger = self.logger.with_context(TAG="test", **context)
        env = env_info()
        spinner.succeed("Test experiment setup completed.")

        # Load first checkpoint to get configuration
        spinner.start("Loading checkpoint configurations...")
        first_checkpoint = self.load(self.config.checkpoint_files[0])

        # Initialize datamodule from first checkpoint config
        # Assuming the checkpoint contains the training config
        checkpoint_config = first_checkpoint.get("config", {})
        self.datamodule = self._create_datamodule_from_checkpoint_config(checkpoint_config)

        # Get test data loader
        test_loader = self.test_loader()
        spinner.succeed("Data loader ready.")

        # Store results for combined plotting
        checkpoint_outputs: t.Dict[str, t.Tuple[torch.Tensor, torch.Tensor]] = {}

        # Test each checkpoint
        for i, checkpoint_path in enumerate(self.config.checkpoint_files):
            spinner.start(f"Testing checkpoint {i + 1}/"
                          f"{len(self.config.checkpoint_files)}: "
                          f"{p_.Path(checkpoint_path).name}")

            checkpoint_result = self.test_ckpt(
                checkpoint_path=checkpoint_path,
                test_loader=test_loader,
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

    def load(self, checkpoint_path: str) -> Checkpoint:
        """Load checkpoint from file."""
        checkpoint_path_obj = p_.Path(checkpoint_path)
        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path_obj, map_location=self.device)
        return checkpoint

    def _create_datamodule_from_checkpoint_config(
            self, checkpoint_config: t.Dict[str, t.Any]) -> ImageDatamodule:
        """Create datamodule using config from checkpoint."""
        # You'll need to reconstruct the config object from the checkpoint
        # This is a simplified version - you might need to adapt based on
        # your actual config structure
        from visflow.resources.config import TrainConfig

        # Create a minimal config for testing
        # You might want to only use relevant parts like data paths, resize,
        # etc.
        config_dict = checkpoint_config.copy()
        config_dict.update({
            "training": {
                "batch_size": self.config.batch_size
            },
            # Override batch size
        })

        # Convert back to config object
        train_config = TrainConfig.from_dict(config_dict)
        return ImageDatamodule(train_config)

    def test_loader(self) -> DataLoader:
        """Get the appropriate test dataloader."""
        if self.config.test_set == "test":
            return self.datamodule.test_dataloader()
        elif self.config.test_set == "val":
            return self.datamodule.val_dataloader()
        else:
            raise ValueError(f"Unknown test set: {self.config.test_set}")

    def test_ckpt(
        self,
        checkpoint_path: str,
        test_loader: DataLoader,
        logger,
        exp_dir: p_.Path,
    ) -> CheckpointTestLog:
        """Test a single checkpoint."""
        # Load checkpoint
        ckpt: Checkpoint = self.load(checkpoint_path)
        checkpoint_name = p_.Path(checkpoint_path).stem

        # Create model
        model = make_model(
            name=ckpt["config"].get("model", {}).get("architecture", "resnet18"),
            pretrained=False,
            num_classes=len(ckpt["classes"]),
            weights_path=None,
        ).to(self.device)

        # Load model weights
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Count parameters
        model_params = sum(p.numel() for p in model.parameters())

        # Run inference
        start_time = time.time()
        test_metrics, test_outputs, test_targets = self.eval(model, test_loader)
        inference_time = time.time() - start_time

        # Generate individual confusion matrix
        class_names = ckpt.get("classes", [])
        cm_path = exp_dir / f"confusion_matrix_{checkpoint_name}.png"
        plot_confusion_matrix(
            confusion_matrix=np.array(test_metrics["confusion_matrix"]),
            class_names=class_names,
            save_path=cm_path,
            show=False,
        )

        # Store outputs for combined plotting
        # You'll need to modify this to actually store the outputs
        # checkpoint_outputs[checkpoint_name] = (test_targets, test_outputs)

        result = CheckpointTestLog(
            checkpoint_path=checkpoint_path,
            checkpoint_name=checkpoint_name,
            epoch=ckpt.get("epoch", 0),
            test_metrics=test_metrics,
            inference_time_sec=inference_time,
            model_params=model_params,
        )

        logger.info(f"Checkpoint {checkpoint_name} tested", **result)
        return result

    def eval(self, model: nn.Module,
             dataloader: DataLoader) -> t.Tuple[Metrics, torch.Tensor, torch.Tensor]:
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

    def plots(self, exp_dir: p_.Path,
              checkpoint_outputs: t.Dict[str, t.Tuple[torch.Tensor, torch.Tensor]]) -> None:
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

    def save(self, exp_dir: p_.Path) -> None:
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
        with open(exp_dir / "detailed_results.json", "w") as f:
            json.dump(self.checkpoint_results, f, indent=2, default=str)
