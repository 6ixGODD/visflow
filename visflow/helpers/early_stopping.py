from __future__ import annotations

import typing as t


class EarlyStopping:
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: t.Literal["min", "max"] = "min",
    ) -> None:
        """
        Early stopping utility for training loops.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: One of "min" or "max". In "min" mode, training will stop when
                the quantity monitored has stopped decreasing; in "max" mode
                it will stop when the quantity has stopped increasing.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None  # type: float | None
        self.early_stop = False

        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        self.mode = mode

        # Set comparison function based on mode
        if mode == "min":
            self.is_better: t.Callable[[float, float], bool] = (
                lambda current, best: current < best - self.min_delta
            )

        else:  # mode == "max"
            self.is_better: t.Callable[[float, float], bool] = (
                lambda current, best: current > best + self.min_delta
            )

    def step(self, score: float, /) -> bool:
        """
        Call this method after each epoch with the validation score.

        Args:
            score: Current validation score (loss, accuracy, f1, etc.)
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.is_better(score, self.best_score):
            # Improvement detected
            self.best_score = score
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

    def reset(self) -> None:
        self.counter = 0
        self.best_score = None
        self.early_stop = False
