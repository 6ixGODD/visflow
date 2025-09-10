from __future__ import annotations

import pathlib as p
import random
import time
import typing as t

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
from visflow.helpers.functional import env_info, MixUpLoss, summary
from visflow.helpers.metrics import compute_metric
from visflow.pipelines import BasePipeline
from visflow.resources.configs import TrainConfig
from visflow.resources.logger import Logger
from visflow.resources.logger.types import LoggingTarget
from visflow.resources.models import BaseClassifier, make_model
from visflow.utils import gen_id, incr_path, seed
from visflow.utils.functional import mixup


class TrainPipeline(BasePipeline):
    __slots__ = (
        '_completed', 'config', 'logger', 'device', 'datamodule', 'best_acc',
        'train_loss_history', 'train_acc_history', 'val_loss_history',
        'val_acc_history', 'best_epoch', 'best_metrics', 'start_time'
    )

    def __init__(self, config: TrainConfig):
        self._completed = False
        self.config = config
        self.logger = Logger(config.logging)
        self.device = torch.device(config.training.device)
        self.datamodule = ImageDatamodule(config)

        self.best_acc = 0.0
        self.best_epoch = 1
        self.best_metrics = None
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.start_time = 0.0

    def __call__(self) -> None:
        self.start_time = time.time()

        # Setup experiment -----------------------------------------------------
        seed(self.config.seed)  # For reproducibility
        exp_id = gen_id(
            pref=self.config.output.experiment_name,
            without_hyphen=True
        )

        output_dir = p.Path(self.config.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        exp_dir = incr_path(output_dir, self.config.output.experiment_name)

        context = Context(
            experiment_id=exp_id,
            experiment_name=self.config.output.experiment_name,
            timestamp=time.strftime("%Y%m%dT%H%M%S", time.localtime())
        )
        display = Display(context)
        logger = self.logger.with_context(**context)
        logger.add_target(
            LoggingTarget(
                logname='stdout',
                loglevel=self.config.logging.loglevel
            ),
            LoggingTarget(logname=exp_dir / '.log.json', loglevel='info'),
        )
        self.config.to_file(fpath=exp_dir / '.config.json')
        env = env_info()

        # Prepare data loaders -------------------------------------------------
        train_loader, val_loader, test_loader = self.datamodule.loaders

        # Initialize model -----------------------------------------------------
        model = make_model(
            name=self.config.model.architecture,
            pretrained=self.config.model.pretrained,
            num_classes=self.config.model.num_classes,
            weights_path=self.config.model.weights_path
        ).to(self.device)

        if isinstance(self.config.resize, tuple):
            x, y = self.config.resize
        else:
            x = y = self.config.resize
        model_summary = summary(model, (3, x, y))

        # Setup loss function
        if self.config.training.label_smoothing > 0:
            criterion = nn.CrossEntropyLoss(
                label_smoothing=self.config.training.label_smoothing
            )
        else:
            criterion = nn.CrossEntropyLoss()
        if self.config.augmentation.mixup.enabled:
            criterion = MixUpLoss(criterion)

        # Setup optimizer ------------------------------------------------------
        if self.config.training.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=self.config.training.momentum,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(
                f"Unsupported optimizer: {self.config.training.optimizer}"
            )

        # Setup learning rate scheduler ----------------------------------------
        if lr_scheduler := self.config.training.lr_scheduler:
            if lr_scheduler == 'step':
                if not self.config.training.step_scheduler:
                    raise ValueError(
                        "StepLR scheduler requires step_scheduler config."
                    )
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=self.config.training.step_scheduler.step_size,
                    gamma=self.config.training.step_scheduler.gamma
                )
            elif lr_scheduler == 'cosine':
                if not self.config.training.cosine_scheduler:
                    raise ValueError(
                        "CosineAnnealingLR scheduler requires "
                        "cosine_scheduler config."
                    )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.config.training.cosine_scheduler.t_max
                )
            elif lr_scheduler == 'plateau':
                if not self.config.training.plateau_scheduler:
                    raise ValueError(
                        "ReduceLROnPlateau scheduler requires "
                        "plateau_scheduler config."
                    )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=self.config.training.plateau_scheduler.mode,
                    patience=self.config.training.plateau_scheduler.patience,
                    factor=self.config.training.plateau_scheduler.factor
                )
            else:
                raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler}")
        else:
            scheduler = None

        # Log experiment start -------------------------------------------------
        startlog = ExperimentStartLog(
            env=env,
            dataset=self.datamodule.info,
            config=self.config.to_dict(),
            model_summary=model_summary,
        )
        display.display_start(startlog)
        logger.info('Experiment started', **startlog)

        # Training loop --------------------------------------------------------
        initial_lr = optimizer.param_groups[0]['lr']

        val_loss, val_metrics = 0.0, None
        for epoch in range(1, self.config.training.epochs + 1):
            epoch_start_time = time.time()

            train_loss, train_acc = self.train(
                model,
                train_loader,
                epoch,
                optimizer,
                criterion,
                logger
            )

            val_loss, val_metrics = self.val(model, val_loader, criterion)
            val_acc = val_metrics['accuracy']

            if scheduler:
                if self.config.training.lr_scheduler == 'plateau':
                    scheduler.step(val_acc)
                else:
                    scheduler.step()

            # Update histories
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)

            # Track best metrics
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_epoch = epoch
                self.best_metrics = val_metrics

            epoch_time = time.time() - epoch_start_time

            # Create epoch log
            epoch_log = EpochLog(
                epoch=epoch,
                total_epochs=self.config.training.epochs,
                avg_metrics=Metrics(loss=train_loss, accuracy=train_acc),
                best_metrics=self.best_metrics,
                epoch_time_sec=epoch_time,
                initial_lr=initial_lr,
                final_lr=optimizer.param_groups[0]['lr'],
            )

            # Display metrics every epoch
            display.display_metrics(epoch_log)

            logger.info(
                f'Epoch {epoch} completed. '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}',
                **epoch_log
            )

        # Test evaluation
        test_metrics = self.test(model, test_loader, criterion)

        # Final experiment log
        total_time = time.time() - self.start_time

        final_metrics = Metrics(
            loss=self.val_loss_history[-1],
            accuracy=self.val_acc_history[-1],
            precision=val_metrics['precision'],
            recall=val_metrics['recall'],
            auc_roc=val_metrics['auc_roc'],
            f1_score=val_metrics['f1_score'],
            confusion_matrix=val_metrics['confusion_matrix'],
        )

        endlog = ExperimentEndLog(
            total_epochs=self.config.training.epochs,
            total_time_sec=total_time,
            final_metrics=final_metrics,
            best_metrics=self.best_metrics,
            test_metrics=test_metrics,
            best_epoch=self.best_epoch,
        )

        display.display_end(endlog)
        logger.info('Training completed', **endlog)

        self._completed = True

    def train(
        self,
        model: BaseClassifier,
        train_loader: torch.utils.data.DataLoader,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        logger: Logger
    ) -> t.Tuple[float, float]:
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for batch, (data, target) in enumerate(train_loader, 1):
            batch_start_time = time.time()
            data, target = data.to(self.device), target.to(self.device)

            if (
                self.config.augmentation.mixup.enabled
                and random.random() < self.config.augmentation.mixup.p
            ):
                mixed_data, target_a, target_b, lam = mixup(
                    data,
                    target,
                    alpha=self.config.augmentation.mixup.alpha
                )

                optimizer.zero_grad()

                forward_start = time.time()
                output = model(mixed_data)
                forward_time = time.time() - forward_start

                loss: torch.Tensor
                if isinstance(criterion, MixUpLoss):
                    loss = criterion(output, target_a, target_b, lam)
                else:
                    loss = (lam *
                            criterion(output, target_a) + (1 - lam) *
                            criterion(output, target_b))

                # Calculate accuracy for mixup (approximate)
                _, pred = torch.max(output, 1)
                correct_a = (pred == target_a).float()  # type: ignore
                correct_b = (pred == target_b).float()  # type: ignore
                batch_acc = ((lam * correct_a + (1 - lam) * correct_b)
                             .mean().item())
            else:
                optimizer.zero_grad()

                forward_start = time.time()
                output = model(data)
                forward_time = time.time() - forward_start

                loss = criterion(output, target)

                # Calculate accuracy
                _, pred = torch.max(output, 1)
                batch_acc = ((pred == target).float()  # type: ignore
                             .mean().item())

            backward_start = time.time()
            loss.backward()

            # Calculate gradient norm
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            gradient_norm = total_norm ** (1. / 2)

            optimizer.step()
            backward_time = time.time() - backward_start

            running_loss += loss.item()
            running_corrects += batch_acc * data.size(0)
            total_samples += data.size(0)

            batch_time = time.time() - batch_start_time
            samples_per_sec = data.size(0) / batch_time

            # GPU memory usage
            gpu_memory_usage = 0.0
            if torch.cuda.is_available():
                gpu_memory_usage = torch.cuda.memory_allocated() / (
                    1024 ** 3)  # GB

            batch_log = BatchLog(
                epoch=epoch,
                total_epochs=self.config.training.epochs,
                batch=batch,
                total_batches=len(train_loader),
                metrics=Metrics(loss=loss.item(), accuracy=batch_acc),
                learning_rate=optimizer.param_groups[0]['lr'],
                gradient_norm=gradient_norm,
                gpu_memory_usage_gb=gpu_memory_usage,
                batch_time_sec=batch_time,
                forward_time_sec=forward_time,
                backward_time_sec=backward_time,
                samples_per_sec=samples_per_sec,
            )

            logger.info(
                f'[Epoch {epoch}/{self.config.training.epochs}] '
                f'Batch {batch}/{len(train_loader)} - '
                f'Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}',
                **batch_log
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_corrects / total_samples
        return epoch_loss, epoch_acc

    def val(
        self,
        model: BaseClassifier,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> t.Tuple[float, Metrics]:
        model.eval()

        val_loss = 0.0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                val_loss += criterion(output, target).item() * data.size(0)

                all_outputs.append(output)
                all_targets.append(target)

        # Concatenate all outputs and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        val_loss /= len(all_targets)

        # Calculate comprehensive metrics
        metrics = compute_metric(
            all_outputs,
            all_targets,
            val_loss,
            num_classes=self.config.model.num_classes
        )

        return val_loss, metrics

    def test(
        self,
        model: BaseClassifier,
        test_loader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> Metrics:
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

        # Concatenate all outputs and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        test_loss /= len(all_targets)

        # Calculate comprehensive metrics
        test_metrics = compute_metric(
            all_outputs,
            all_targets,
            test_loss,
            num_classes=self.config.model.num_classes
        )

        return test_metrics

    def plots(
        self,
        logger: Logger,
    ) -> None:
        pass

    def save(self) -> None:
        pass
