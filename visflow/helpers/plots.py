from __future__ import annotations

import pathlib as p
import typing as t

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import auc, roc_curve

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12


def plot_training_curves(
    train_loss_history: t.List[float],
    val_loss_history: t.List[float],
    train_acc_history: t.List[float],
    val_acc_history: t.List[float],
    save_path: p.Path | str | None = None
) -> None:
    epochs = range(1, len(train_loss_history) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Loss curves
    ax1.plot(
        epochs,
        train_loss_history,
        'b-',
        label='Training Loss',
        linewidth=2
    )
    ax1.plot(
        epochs,
        val_loss_history,
        'r-',
        label='Validation Loss',
        linewidth=2
    )
    ax1.set_title('Training and Validation Loss', fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(
        epochs,
        train_acc_history,
        'b-',
        label='Training Accuracy',
        linewidth=2
    )
    ax2.plot(
        epochs,
        val_acc_history,
        'r-',
        label='Validation Accuracy',
        linewidth=2
    )
    ax2.set_title('Training and Validation Accuracy', fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_roc_curve(
    y_true: np.ndarray | torch.Tensor,
    y_scores: np.ndarray | torch.Tensor,
    num_classes: int,
    class_names: t.List[str] | None = None,
    save_path: p.Path | str | None = None
) -> None:
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()

    # Convert to one-hot if needed
    if y_true.ndim == 1:
        y_true_onehot = np.eye(num_classes)[y_true]
    else:
        y_true_onehot = y_true

    # If y_scores is logits, convert to probabilities
    if y_scores.max() > 1.0:
        y_scores = torch.softmax(torch.from_numpy(y_scores), dim=1).numpy()

    plt.figure(figsize=(10, 8))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)

        class_label = class_names[i] if class_names else f'Class {i}'
        plt.plot(
            fpr,
            tpr,
            color=plt.cm.get_cmap('tab10')(i),
            linewidth=2,
            label=f'{class_label} (AUC = {roc_auc:.3f})'
        )

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Multi-class Classification', fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: t.List[str] | None = None,
    normalize: bool = True,
    save_path: p.Path | str | None = None
) -> None:
    if normalize:
        cm = (confusion_matrix.astype('float') /
              confusion_matrix.sum(axis=1)[:, np.newaxis])
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        cm = confusion_matrix
        fmt = 'd'
        title = 'Confusion Matrix'

    plt.figure(figsize=(10, 8))

    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(confusion_matrix))]

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(title, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_all_metrics(
    train_loss_history: t.List[float],
    val_loss_history: t.List[float],
    train_acc_history: t.List[float],
    val_acc_history: t.List[float],
    test_outputs: torch.Tensor,
    test_targets: torch.Tensor,
    confusion_matrix: np.ndarray,
    num_classes: int,
    class_names: t.List[str] | None = None,
    save_dir: p.Path | str | None = None
) -> None:
    if save_dir:
        save_dir = p.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Plot training curves
        plot_training_curves(
            train_loss_history,
            val_loss_history,
            train_acc_history,
            val_acc_history,
            save_path=save_dir / 'training_curves.png'
        )

        # Plot ROC curves
        plot_roc_curve(
            test_targets,
            test_outputs,
            num_classes,
            class_names=class_names,
            save_path=save_dir / 'roc_curves.png'
        )

        # Plot confusion matrix
        plot_confusion_matrix(
            confusion_matrix,
            class_names=class_names,
            save_path=save_dir / 'confusion_matrix.png'
        )

        # Plot normalized confusion matrix
        plot_confusion_matrix(
            confusion_matrix,
            class_names=class_names,
            normalize=True,
            save_path=save_dir / 'confusion_matrix_normalized.png'
        )
    else:
        plot_training_curves(
            train_loss_history,
            val_loss_history,
            train_acc_history,
            val_acc_history
        )
        plot_roc_curve(test_targets, test_outputs, num_classes, class_names)
        plot_confusion_matrix(confusion_matrix, class_names)
        plot_confusion_matrix(confusion_matrix, class_names, normalize=True)
