from __future__ import annotations

import pathlib as p
import typing as t

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12


def plot_training_curves(train_loss_history: t.List[float],
                         val_loss_history: t.List[float],
                         train_acc_history: t.List[float],
                         val_acc_history: t.List[float],
                         save_path: p.Path | None = None,
                         show: bool = True) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss curves
    epochs = range(1, len(train_loss_history) + 1)
    ax1.plot(epochs, train_loss_history, "b-", label="Training Loss", linewidth=2)
    ax1.plot(epochs, val_loss_history, "r-", label="Validation Loss", linewidth=2)
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy curves
    ax2.plot(epochs, train_acc_history, "b-", label="Training Accuracy", linewidth=2)
    ax2.plot(epochs, val_acc_history, "r-", label="Validation Accuracy", linewidth=2)
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(y_true: torch.Tensor,
                   y_pred_probs: torch.Tensor,
                   num_classes: int,
                   class_names: t.List[str] | None = None,
                   save_path: p.Path | None = None,
                   show: bool = True) -> t.Dict[str, float]:
    # Convert to numpy
    y_true_np = y_true.cpu().numpy()
    y_pred_probs_np = y_pred_probs.cpu().numpy()

    # For multi-class, binarize the output
    if num_classes > 2:
        y_true_bin = label_binarize(y_true_np, classes=range(num_classes))

        plt.figure(figsize=(10, 8))

        auc_scores = {}
        colors = plt.cm.Set1(np.linspace(0, 1, num_classes))  # type: ignore

        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs_np[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores[f"class_{i}"] = roc_auc

            class_name = class_names[i] if class_names else f"Class {i}"
            plt.plot(
                fpr,
                tpr,
                color=colors[i],
                linewidth=2,
                label=f"{class_name} (AUC = {roc_auc:.3f})",
            )

        # Calculate macro-average ROC curve and AUC
        all_fpr = np.unique(
            np.concatenate([
                roc_curve(y_true_bin[:, i], y_pred_probs_np[:, i])[0] for i in range(num_classes)
            ]))
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs_np[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)

        mean_tpr /= num_classes
        macro_auc = auc(all_fpr, mean_tpr)
        auc_scores["macro_avg"] = macro_auc

        plt.plot(
            all_fpr,
            mean_tpr,
            color="navy",
            linestyle="--",
            linewidth=2,
            label=f"Macro-average (AUC = {macro_auc:.3f})",
        )

    else:
        # Binary classification
        if y_pred_probs_np.shape[1] == 2:
            # Use probability of positive class
            y_pred_probs_binary = y_pred_probs_np[:, 1]
        else:
            y_pred_probs_binary = y_pred_probs_np.flatten()

        fpr, tpr, _ = roc_curve(y_true_np, y_pred_probs_binary)
        roc_auc = auc(fpr, tpr)
        auc_scores = {"binary": roc_auc}

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            linewidth=2,
            label=f"ROC curve (AUC = {roc_auc:.3f})",
        )

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

    return auc_scores


def plot_combined_roc_curves(checkpoint_results: t.Dict[str, t.Tuple[torch.Tensor, torch.Tensor]],
                             num_classes: int,
                             class_names: t.List[str] | None = None,
                             save_path: p.Path | None = None,
                             show: bool = True) -> t.Dict[str, t.Dict[str, float]]:
    """
    Plot ROC curves for multiple checkpoints on the same figure.

    Args:
        checkpoint_results: Dict mapping checkpoint names to (y_true, y_pred_probs) tuples
        num_classes: Number of classes
        class_names: Optional list of class names
        save_path: Path to save the plot
        show: Whether to show the plot

    Returns:
        Dictionary mapping checkpoint names to their AUC scores
    """
    if num_classes > 2:
        # Multi-class ROC curves
        fig, axes = plt.subplots(2, (num_classes + 1) // 2, figsize=(15, 10))
        if num_classes == 1:
            axes = [axes]
        elif (num_classes + 1) // 2 == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        all_auc_scores = {}
        colors = plt.cm.Set1(np.linspace(0, 1, len(checkpoint_results)))  # type: ignore

        # Plot for each class
        for class_idx in range(num_classes):
            ax = axes[class_idx]
            class_name = class_names[class_idx] if class_names else f"Class {class_idx}"

            for i, (checkpoint_name, (y_true,
                                      y_pred_probs)) in enumerate(checkpoint_results.items()):
                y_true_np = y_true.cpu().numpy()
                y_pred_probs_np = y_pred_probs.cpu().numpy()

                # Binarize for this class
                y_true_bin = label_binarize(y_true_np, classes=range(num_classes))

                fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], y_pred_probs_np[:, class_idx])
                roc_auc = auc(fpr, tpr)

                if checkpoint_name not in all_auc_scores:
                    all_auc_scores[checkpoint_name] = {}
                all_auc_scores[checkpoint_name][f"class_{class_idx}"] = roc_auc

                ax.plot(fpr,
                        tpr,
                        color=colors[i],
                        linewidth=2,
                        label=f"{checkpoint_name} (AUC={roc_auc:.3f})")

            ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve - {class_name}")
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for i in range(num_classes, len(axes)):
            fig.delaxes(axes[i])

    else:
        # Binary classification
        plt.figure(figsize=(10, 8))

        all_auc_scores = {}
        colors = plt.cm.Set1(np.linspace(0, 1, len(checkpoint_results)))  # type: ignore

        for i, (checkpoint_name, (y_true, y_pred_probs)) in enumerate(checkpoint_results.items()):
            y_true_np = y_true.cpu().numpy()
            y_pred_probs_np = y_pred_probs.cpu().numpy()

            if y_pred_probs_np.shape[1] == 2:
                y_pred_probs_binary = y_pred_probs_np[:, 1]
            else:
                y_pred_probs_binary = y_pred_probs_np.flatten()

            fpr, tpr, _ = roc_curve(y_true_np, y_pred_probs_binary)
            roc_auc = auc(fpr, tpr)
            all_auc_scores[checkpoint_name] = {"binary": roc_auc}

            plt.plot(fpr,
                     tpr,
                     color=colors[i],
                     linewidth=2,
                     label=f"{checkpoint_name} (AUC={roc_auc:.3f})")

        plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves Comparison")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return all_auc_scores


def plot_confusion_matrix(confusion_matrix: np.ndarray,
                          class_names: t.List[str] | None = None,
                          normalize: bool = False,
                          save_path: p.Path | None = None,
                          show: bool = True) -> None:
    if normalize:
        confusion_matrix = (confusion_matrix.astype("float") /
                            confusion_matrix.sum(axis=1)[:, np.newaxis])
        title = "Normalized Confusion Matrix"
        fmt = ".2f"
    else:
        title = "Confusion Matrix"
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion_matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(confusion_matrix.shape[0])]

    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel="True Label",
           xlabel="Predicted Label")

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = confusion_matrix.max() / 2.0
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j,
                    i,
                    format(confusion_matrix[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()
