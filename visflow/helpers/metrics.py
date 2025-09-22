from __future__ import annotations

import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from visflow.context import Metrics


def compute_metric(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    loss: float | None = None,
    num_classes: int | None = None,
) -> Metrics:
    with torch.no_grad():
        # Get predictions
        _, preds = torch.max(outputs, 1)

        # Convert to numpy for sklearn metrics
        y_true = targets.cpu().numpy()
        y_pred = preds.cpu().numpy()
        y_prob = torch.softmax(outputs, dim=1).cpu().numpy()

        # Accuracy
        accuracy = (preds == targets).float().mean().item()  # type: ignore

        # Precision, Recall, F1
        if num_classes and num_classes > 2:
            # Multi-class
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )
            # AUC-ROC for multi-class (one-vs-rest)
            try:
                auc_roc = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="weighted"
                )
            except ValueError:
                auc_roc = None
        else:
            # Binary classification
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            # AUC-ROC for binary
            try:
                auc_roc = roc_auc_score(
                    y_true, y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob
                )
            except (ValueError, IndexError):
                auc_roc = None

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)

        if num_classes and num_classes > 2:
            # Multi-class: calculate macro-averaged sensitivity and specificity
            sensitivities = []
            specificities = []

            for i in range(num_classes):
                # For each class, calculate sensitivity and specificity
                tp = cm[i, i]  # True positives for class i
                fn = cm[i, :].sum() - tp  # False negatives for class i
                fp = cm[:, i].sum() - tp  # False positives for class i
                tn = cm.sum() - tp - fn - fp  # True negatives for class i

                # Sensitivity (Recall/True Positive Rate) = TP / (TP + FN)
                class_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                sensitivities.append(class_sensitivity)

                # Specificity (True Negative Rate) = TN / (TN + FP)
                class_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                specificities.append(class_specificity)

            # Macro average
            sensitivity = (
                sum(sensitivities) / len(sensitivities) if sensitivities else 0.0
            )
            specificity = (
                sum(specificities) / len(specificities) if specificities else 0.0
            )

        else:
            # Binary classification
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()

                # Sensitivity (Recall/True Positive Rate) = TP / (TP + FN)
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                # Specificity (True Negative Rate) = TN / (TN + FP)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                # Handle edge case where only one class is present
                sensitivity = 0.0
                specificity = 0.0

        return Metrics(
            loss=loss,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            auc_roc=auc_roc,
            f1_score=f1,
            confusion_matrix=cm.tolist(),
            sensitivity=sensitivity,
            specificity=specificity,
        )
