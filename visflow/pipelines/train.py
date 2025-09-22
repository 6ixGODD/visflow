from __future__ import annotations

import pathlib as p
import random
import time
import typing as t

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from visflow.context import (
    BatchLog,
    Context,
    EpochLog,
    ExperimentEndLog,
    ExperimentStartLog,
    Metrics,
)
from visflow.data import ImageDatamodule
from visflow.helpers.display import Display
from visflow.helpers.early_stopping import EarlyStopping
from visflow.helpers.functional import env_info, MixUpLoss, summary
from visflow.helpers.metrics import compute_metric
from visflow.helpers.plotting import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
)
from visflow.pipelines import BasePipeline
from visflow.resources.config import TrainConfig
from visflow.resources.logger import _LoggerContext, Logger
from visflow.resources.logger.types import LoggingTarget
from visflow.resources.models import BaseClassifier, make_model
from visflow.types import Checkpoint
from visflow.utils import gen_id, incr_path, seed, spinner
from visflow.utils.functional import mixup


class TrainPipeline(BasePipeline):
    __slots__ = BasePipeline.__slots__ + (
        "config",
        "logger",
        "device",
        "datamodule",
        "best_acc",
        "train_loss_history",
        "train_acc_history",
        "val_loss_history",
        "val_acc_history",
        "best_epoch",
        "best_metrics",
        "final_metrics",
        "start_time",
        "best_val_outputs",
        "best_val_targets",
        "final_val_outputs",
        "final_val_targets",
    )

    def __init__(self, config: TrainConfig):
        self._completed = False
        self.config = config
        self.logger = Logger(config.logging)
        self.device = torch.device(config.training.device)
        self.datamodule = ImageDatamodule(config)

        self.best_acc = 0.0
        self.best_epoch = 1
        self.best_metrics = None  # type: Metrics | None
        self.final_metrics = None  # type: Metrics | None
        self.train_loss_history = []  # type: t.List[float]
        self.train_acc_history = []  # type: t.List[float]
        self.val_loss_history = []  # type: t.List[float]
        self.val_acc_history = []  # type: t.List[float]
        self.start_time = 0.0

        # Store outputs and targets for plotting
        self.best_val_outputs = None  # type: torch.Tensor | None
        self.best_val_targets = None  # type: torch.Tensor | None
        self.final_val_outputs = None  # type: torch.Tensor | None
        self.final_val_targets = None  # type: torch.Tensor | None

    def __call__(self) -> None:
        self.start_time = time.time()

        # Setup experiment -----------------------------------------------------
        spinner.start("Setting up experiment...")
        seed(self.config.seed)
        exp_id = gen_id(
            pref=self.config.output.experiment_name,
            without_hyphen=True
        )

        output_dir = p.Path(self.config.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        exp_name = (self.config.output.experiment_name
                    if self.config.output.experiment_name != 'auto'
                    else self.config.model.architecture)
        exp_dir = incr_path(output_dir, exp_name)

        context = Context(
            experiment_id=exp_id,
            experiment_name=exp_name,
            timestamp=time.strftime("%Y%m%dT%H%M%S", time.localtime()),
        )
        display = Display(context)
        self.logger.add_target(
            LoggingTarget(logname=exp_dir / ".log.json", loglevel="info"),
        )
        logger = self.logger.with_context(TAG="train", **context)
        self.config.to_file(fpath=exp_dir / ".config.json")
        env = env_info()
        spinner.succeed("Experiment setup completed.")

        # Prepare data loaders -------------------------------------------------
        spinner.start("Preparing data loaders...")
        train_loader, val_loader, test_loader = self.datamodule.loaders
        spinner.succeed("Data loaders ready.")

        # Initialize model -----------------------------------------------------
        spinner.start("Initializing model...")
        model = make_model(
            name=self.config.model.architecture,
            pretrained=self.config.model.pretrained,
            num_classes=self.config.model.num_classes,
            weights_path=self.config.model.weights_path,
        ).to(self.device)

        size = self.config.resize.size
        if isinstance(size, tuple):
            x, y = size
        else:
            x = y = size
        model_summary = summary(model, (3, x, y))
        spinner.succeed("Model initialized.")

        # Setup loss function and optimizer ------------------------------------
        criterion = self._setup_criterion()
        optimizer = self._setup_optimizer(model)
        scheduler = self._setup_scheduler(optimizer)

        # Log experiment start -------------------------------------------------
        startlog = ExperimentStartLog(
            env=env,
            dataset=self.datamodule.info,
            config=self.config.to_dict(),
            model_summary=model_summary,
        )
        display.display_start(startlog)
        logger.info("Experiment started", **startlog)

        # Training loop --------------------------------------------------------
        initial_lr = optimizer.param_groups[0]["lr"]
        checkpoint_freq = getattr(self.config.output, "checkpoint_frequency", 0)

        early_stopping = EarlyStopping(
            patience=self.config.training.early_stopping_patience,
            min_delta=self.config.training.early_stopping_min_delta,
            mode=(  # type: ignore
                "min" if self.config.training.early_stopping_target == "loss"
                else "max"
            ),
        )

        val_acc = 0.0  # Initialize val_acc for first epoch
        epoch = 0  # cache epoch for saving final checkpoint
        for epoch in range(1, self.config.training.epochs + 1):
            epoch_start_time = time.time()

            # Training
            train_loss, train_acc = self.train(
                model, train_loader, epoch, optimizer, criterion, logger
            )

            # Validation
            val_loss, val_metrics, val_outputs, val_targets = self.val(
                model, val_loader, criterion
            )
            val_acc = val_metrics["accuracy"]

            # Update scheduler
            self._update_scheduler(scheduler, val_acc)

            # Update histories
            self._update_histories(train_loss, train_acc, val_loss, val_acc)

            # Track best and final metrics
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_epoch = epoch
                self.best_metrics = val_metrics
                self.best_val_outputs = val_outputs
                self.best_val_targets = val_targets
                # Save best checkpoint
                self.save(
                    logger,
                    exp_dir,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    val_acc,
                    "best"
                )

            # Always update final metrics (last epoch)
            self.final_metrics = val_metrics
            self.final_val_outputs = val_outputs
            self.final_val_targets = val_targets

            # Save checkpoint based on frequency
            if checkpoint_freq > 0 and epoch % checkpoint_freq == 0:
                self.save(
                    logger,
                    exp_dir,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    val_acc,
                    "frequent",
                )

            epoch_time = time.time() - epoch_start_time

            # Log epoch results
            epoch_log = EpochLog(
                epoch=epoch,
                total_epochs=self.config.training.epochs,
                train_metrics=Metrics(loss=train_loss, accuracy=train_acc),
                val_metrics=val_metrics,
                best_val_metrics=self.best_metrics,
                best_epoch=self.best_epoch,
                epoch_time_sec=epoch_time,
                initial_lr=initial_lr,
                final_lr=optimizer.param_groups[0]["lr"],
            )

            display.display_metrics(epoch_log)
            logger.info(
                f"Epoch {epoch} completed. "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}",
                **epoch_log,
            )
            # Check early stopping
            if self.config.training.early_stopping_target == "loss":
                score = val_loss
            elif self.config.training.early_stopping_target == "accuracy":
                score = val_acc
            elif self.config.training.early_stopping_target == "f1":
                score = val_metrics["f1_score"]
            elif self.config.training.early_stopping_target == "precision":
                score = val_metrics["precision"]
            elif self.config.training.early_stopping_target == "recall":
                score = val_metrics["recall"]
            else:
                raise ValueError(
                    f"Unsupported early_stopping_target: "
                    f"{self.config.training.early_stopping_target}"
                )
            if early_stopping.step(score):
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"No improvement in "
                    f"{self.config.training.early_stopping_target} for "
                    f"{self.config.training.early_stopping_patience} epochs."
                )
                break
        # Save final checkpoint
        self.save(
            logger,
            exp_dir,
            epoch,
            model,
            optimizer,
            scheduler,
            val_acc,
            "final"
        )

        # Test evaluation ------------------------------------------------------
        test_metrics, test_outputs, test_targets = self.test(
            model, test_loader, criterion
        )

        # Generate plots -------------------------------------------------------
        class_names = self.datamodule.classes
        self.plots(
            logger,
            exp_dir,
            class_names,
            test_outputs,
            test_targets,
            test_metrics
        )

        # Save comprehensive metrics -------------------------------------------
        self.save_comprehensive_metrics(logger, exp_dir, test_metrics)

        # Final experiment log -------------------------------------------------
        total_time = time.time() - self.start_time
        endlog = ExperimentEndLog(
            total_epochs=self.config.training.epochs,
            total_time_sec=total_time,
            final_metrics=t.cast(Metrics, self.final_metrics),
            best_metrics=t.cast(Metrics, self.best_metrics),
            test_metrics=test_metrics,
            best_epoch=self.best_epoch,
        )

        display.display_end(endlog)
        logger.info("Training completed", **endlog)
        self._completed = True

    def _setup_criterion(self) -> nn.Module:
        """Setup loss function."""
        criterion: nn.Module
        if self.config.training.label_smoothing > 0:
            criterion = nn.CrossEntropyLoss(
                label_smoothing=self.config.training.label_smoothing
            )
        else:
            criterion = nn.CrossEntropyLoss()

        if self.config.augmentation.mixup.enabled:
            criterion = MixUpLoss(criterion)

        return criterion

    def _setup_optimizer(self, model: BaseClassifier) -> torch.optim.Optimizer:
        """Setup optimizer."""
        optimizer_name = self.config.training.optimizer.lower()
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay

        if optimizer_name == "adam":
            return torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "adamw":
            return torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=self.config.training.momentum,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _setup_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Setup learning rate scheduler."""
        lr_scheduler = self.config.training.lr_scheduler
        if not lr_scheduler:
            return None

        if lr_scheduler == "step":
            if not self.config.training.step_scheduler:
                raise ValueError(
                    "StepLR scheduler requires step_scheduler config."
                )
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.training.step_scheduler.step_size,
                gamma=self.config.training.step_scheduler.gamma,
            )
        elif lr_scheduler == "cosine":
            if not self.config.training.cosine_scheduler:
                raise ValueError(
                    "CosineAnnealingLR scheduler requires cosine_scheduler "
                    "config."
                )
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.training.cosine_scheduler.t_max
            )
        elif lr_scheduler == "plateau":
            if not self.config.training.plateau_scheduler:
                raise ValueError(
                    "ReduceLROnPlateau scheduler requires plateau_scheduler "
                    "config."
                )
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.config.training.plateau_scheduler.mode,
                patience=self.config.training.plateau_scheduler.patience,
                factor=self.config.training.plateau_scheduler.factor,
            )
        else:
            raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler}")

    def _update_scheduler(
        self,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        val_acc: float
    ) -> None:
        """Update learning rate scheduler."""
        if scheduler:
            if self.config.training.lr_scheduler == "plateau" and isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step(val_acc)
            else:
                scheduler.step()

    def _update_histories(
        self,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float
    ) -> None:
        """Update training histories."""
        self.train_loss_history.append(train_loss)
        self.train_acc_history.append(train_acc)
        self.val_loss_history.append(val_loss)
        self.val_acc_history.append(val_acc)

    def train(
        self,
        model: BaseClassifier,
        train_loader: torch.utils.data.DataLoader[torch.Tensor],
        epoch: int,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        logger: _LoggerContext,
    ) -> t.Tuple[float, float]:
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for batch, (data, target) in enumerate(train_loader, 1):
            batch_start_time = time.time()
            data, target = (
                data.to(self.device),  # type: ignore
                target.to(self.device),
            )

            # Handle mixup augmentation
            use_mixup = (
                self.config.augmentation.mixup.enabled
                and random.random() < self.config.augmentation.mixup.p
            )

            if use_mixup:
                loss, batch_acc = self._train_step_mixup(
                    self.config.augmentation.mixup.alpha,
                    data,
                    target,
                    model,
                    optimizer,
                    criterion,
                )
            else:
                loss, batch_acc = self._train_step(
                    data, target, model, optimizer, criterion
                )

            # Update metrics
            running_loss += loss.item() * data.size(0)
            running_corrects += batch_acc * data.size(0)
            total_samples += data.size(0)

            # Log batch information
            self._log_batch(
                batch,
                epoch,
                loss,
                batch_acc,
                model,
                optimizer,
                data,
                batch_start_time,
                train_loader,
                logger,
            )

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        return epoch_loss, epoch_acc

    @staticmethod
    def _train_step_mixup(
        alpha: float,
        data: torch.Tensor,
        target: torch.Tensor,
        model: BaseClassifier,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> t.Tuple[torch.Tensor, float]:
        mixed_data, target_a, target_b, lam = mixup(data, target, alpha=alpha)

        optimizer.zero_grad()
        output = model(mixed_data)

        if isinstance(criterion, MixUpLoss):
            loss = criterion(output, target_a, target_b, lam)
        else:
            loss = lam * criterion(output, target_a) + (1 - lam) * criterion(
                output, target_b
            )

        loss.backward()
        optimizer.step()

        # Calculate approximate accuracy for mixup
        _, pred = torch.max(output, 1)
        correct_a = (pred == target_a).float()  # type: ignore
        correct_b = (pred == target_b).float()  # type: ignore
        batch_acc = (lam * correct_a + (1 - lam) * correct_b).mean().item()

        return loss, batch_acc

    @staticmethod
    def _train_step(
        data: torch.Tensor,
        target: torch.Tensor,
        model: BaseClassifier,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> t.Tuple[torch.Tensor, float]:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        _, pred = torch.max(output, 1)
        batch_acc = (pred == target).float().mean().item()  # type: ignore

        return loss, batch_acc

    def _log_batch(
        self,
        batch: int,
        epoch: int,
        loss: torch.Tensor,
        batch_acc: float,
        model: BaseClassifier,
        optimizer: torch.optim.Optimizer,
        data: torch.Tensor,
        batch_start_time: float,
        train_loader: torch.utils.data.DataLoader[torch.Tensor],
        logger: _LoggerContext,
    ) -> None:
        """Log batch information."""
        batch_time = time.time() - batch_start_time

        # Calculate gradient norm
        total_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in model.parameters()
            if p.grad is not None
        )
        gradient_norm = total_norm ** 0.5

        # GPU memory usage
        gpu_memory_usage = (
            torch.cuda.memory_allocated() / (1024 ** 3)
            if torch.cuda.is_available()
            else 0.0
        )

        batch_log = BatchLog(
            epoch=epoch,
            total_epochs=self.config.training.epochs,
            batch=batch,
            total_batches=len(train_loader),
            metrics=Metrics(loss=loss.item(), accuracy=batch_acc),
            learning_rate=optimizer.param_groups[0]["lr"],
            gradient_norm=gradient_norm,
            gpu_memory_usage_gb=gpu_memory_usage,
            batch_time_sec=batch_time,
            forward_time_sec=0.0,  # Could be tracked separately if needed
            backward_time_sec=0.0,  # Could be tracked separately if needed
            samples_per_sec=data.size(0) / batch_time,
        )

        logger.info(
            f"[Epoch {epoch}/{self.config.training.epochs}] "
            f"Batch {batch}/{len(train_loader)} - "
            f"Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}",
            **batch_log,
        )

    def val(
        self,
        model: BaseClassifier,
        val_loader: torch.utils.data.DataLoader[torch.Tensor],
        criterion: nn.Module,
    ) -> t.Tuple[float, Metrics, torch.Tensor, torch.Tensor]:
        model.eval()
        val_loss = 0.0
        total_samples = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                val_loss += criterion(output, target).item() * data.size(0)
                total_samples += data.size(0)
                all_outputs.append(output)
                all_targets.append(target)

        all_outputs_tensor = torch.cat(all_outputs, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        val_loss /= total_samples

        metrics = compute_metric(
            all_outputs_tensor,
            all_targets_tensor,
            val_loss,
            num_classes=self.config.model.num_classes,
        )
        return val_loss, metrics, all_outputs_tensor, all_targets_tensor

    def test(
        self,
        model: BaseClassifier,
        test_loader: torch.utils.data.DataLoader[torch.Tensor],
        criterion: nn.Module,
    ) -> t.Tuple[Metrics, torch.Tensor, torch.Tensor]:
        """Test the model and return metrics, outputs, and targets."""
        spinner.start("Evaluating on test set...")
        model.eval()
        test_loss = 0.0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += criterion(output, target).item() * data.size(0)
                all_outputs.append(output)
                all_targets.append(target)

        all_outputs_tensor = torch.cat(all_outputs, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        test_loss /= len(all_targets)

        test_metrics = compute_metric(
            all_outputs_tensor,
            all_targets_tensor,
            test_loss,
            num_classes=self.config.model.num_classes,
        )
        spinner.succeed("Test evaluation completed.")
        return test_metrics, all_outputs_tensor, all_targets_tensor

    def plots(
        self,
        logger: _LoggerContext,
        exp_dir: p.Path,
        class_names: t.List[str],
        test_outputs: torch.Tensor,
        test_targets: torch.Tensor,
        test_metrics: Metrics,
    ) -> None:
        """Generate and save all plots."""
        plots_dir = exp_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Plot training curves
        plot_training_curves(
            train_loss_history=self.train_loss_history,
            val_loss_history=self.val_loss_history,
            train_acc_history=self.train_acc_history,
            val_acc_history=self.val_acc_history,
            save_path=plots_dir / "training_curves.png",
            show=False,
        )

        # Plot ROC curves for validation (best), validation (final), and test
        if self.best_val_outputs is not None:
            plot_roc_curve(
                y_true=(
                    self.best_val_targets
                    if self.best_val_targets is not None
                    else torch.tensor([])
                ),
                y_pred_probs=torch.softmax(self.best_val_outputs, dim=1),
                num_classes=self.config.model.num_classes,
                class_names=class_names,
                save_path=plots_dir / "roc_curve_best_val.png",
                show=False,
            )

        if self.final_val_outputs is not None:
            plot_roc_curve(
                y_true=(
                    self.final_val_targets
                    if self.final_val_targets is not None
                    else torch.tensor([])
                ),
                y_pred_probs=torch.softmax(self.final_val_outputs, dim=1),
                num_classes=self.config.model.num_classes,
                class_names=class_names,
                save_path=plots_dir / "roc_curve_final_val.png",
                show=False,
            )

        if test_outputs is not None and test_targets is not None:
            plot_roc_curve(
                y_true=test_targets,
                y_pred_probs=torch.softmax(test_outputs, dim=1),
                num_classes=self.config.model.num_classes,
                class_names=class_names,
                save_path=plots_dir / "roc_curve_test.png",
                show=False,
            )

        # Plot confusion matrices
        self._plot_confusion_matrices(plots_dir, class_names, test_metrics)

        logger.info(f"Plots saved to: {plots_dir}")

    def _plot_confusion_matrices(
        self, plots_dir: p.Path, class_names: t.List[str], test_metrics: Metrics
    ) -> None:
        """Plot confusion matrices for different datasets."""
        # Test confusion matrix
        if test_metrics:
            cm = np.array(test_metrics["confusion_matrix"])

            plot_confusion_matrix(
                confusion_matrix=cm,
                class_names=class_names,
                normalize=False,
                save_path=plots_dir / "confusion_matrix_test.png",
                show=False,
            )

            plot_confusion_matrix(
                confusion_matrix=cm,
                class_names=class_names,
                normalize=True,
                save_path=plots_dir / "confusion_matrix_test_normalized.png",
                show=False,
            )

        # Best validation confusion matrix
        if self.best_metrics:
            cm = np.array(self.best_metrics["confusion_matrix"])

            plot_confusion_matrix(
                confusion_matrix=cm,
                class_names=class_names,
                normalize=False,
                save_path=plots_dir / "confusion_matrix_best_val.png",
                show=False,
            )

            plot_confusion_matrix(
                confusion_matrix=cm,
                class_names=class_names,
                normalize=True,
                save_path=(
                    plots_dir / "confusion_matrix_best_val_normalized.png"
                ),
                show=False,
            )

        # Final validation confusion matrix
        if self.final_metrics:
            cm = np.array(self.final_metrics["confusion_matrix"])

            plot_confusion_matrix(
                confusion_matrix=cm,
                class_names=class_names,
                normalize=False,
                save_path=plots_dir / "confusion_matrix_final_val.png",
                show=False,
            )

            plot_confusion_matrix(
                confusion_matrix=cm,
                class_names=class_names,
                normalize=True,
                save_path=(
                    plots_dir /
                    "confusion_matrix_final_val_normalized.png"
                ),
                show=False,
            )

    def save(
        self,
        logger: _LoggerContext,
        exp_dir: p.Path,
        epoch: int,
        model: BaseClassifier,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        accuracy: float,
        mode: t.Literal["best", "final", "frequent"],
    ) -> None:
        """Save model checkpoint."""
        ckpt_dir = exp_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        if mode == "best":
            # Remove previous best checkpoints
            for f in ckpt_dir.glob("best_epoch*.pth"):
                f.unlink(missing_ok=True)
            fname = f"best_epoch_{epoch}"
        elif mode == "final":
            fname = f"final_epoch_{epoch}"
        elif mode == "frequent":
            fname = f"epoch_{epoch}"
        else:
            raise ValueError(f"Unsupported save mode: {mode}")

        ckpt = Checkpoint(
            epoch=epoch,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict() if scheduler else None,
            accuracy=accuracy,
            config=self.config.to_dict(),
            classes=self.datamodule.classes,
            class_to_idx=self.datamodule.class_to_idx,
        )

        checkpoint_path = ckpt_dir / f"{fname}.pth"
        torch.save(ckpt, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def save_comprehensive_metrics(
        self,
        logger: _LoggerContext,
        exp_dir: p.Path,
        test_metrics: Metrics,
    ) -> None:
        """Save comprehensive metrics including test, best validation,
        and final validation."""
        metric_dir = exp_dir / "metrics"
        metric_dir.mkdir(exist_ok=True)

        # Prepare metrics data
        metrics_data = []

        # Test metrics
        if test_metrics:
            test_row = {
                "dataset": "test",
                "architecture": self.config.model.architecture,
                "epoch": self.config.training.epochs,
                **{k: v for k, v in test_metrics.items() if
                   k != "confusion_matrix"},
                **{f"cm_{i}": v for i, v in
                   enumerate(
                       np.array(test_metrics["confusion_matrix"]).flatten()
                   )}
            }
            metrics_data.append(test_row)

        # Best validation metrics
        if self.best_metrics:
            best_val_row = {
                "dataset": "validation_best",
                "architecture": self.config.model.architecture,
                "epoch": self.best_epoch,
                **{
                    k: v
                    for k, v in self.best_metrics.items()
                    if k != "confusion_matrix"
                },
            }
            metrics_data.append(best_val_row)

        # Final validation metrics
        if self.final_metrics:
            final_val_row = {
                "dataset": "validation_final",
                "architecture": self.config.model.architecture,
                "epoch": self.config.training.epochs,
                **{
                    k: v
                    for k, v in self.final_metrics.items()
                    if k != "confusion_matrix"
                },
            }
            metrics_data.append(final_val_row)

        # Save to CSV
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            metrics_path = metric_dir / "comprehensive_metrics.csv"
            df.to_csv(metrics_path, index=False)
            logger.info(f"Comprehensive metrics saved: {metrics_path}")

        # Save individual metrics files for compatibility
        for data in metrics_data:
            dataset_name = data["dataset"]
            dataset_metrics = {k: v for k, v in data.items() if k != "dataset"}
            df_single = pd.DataFrame([dataset_metrics])
            single_path = metric_dir / f"metrics_{dataset_name}.csv"
            df_single.to_csv(single_path, index=False)
            logger.info(f"Individual metrics saved: {single_path}")
