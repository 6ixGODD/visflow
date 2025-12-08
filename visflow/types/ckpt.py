from __future__ import annotations

import typing as t

__vf_ckpt_version__ = 1


class Checkpoint(t.NamedTuple):
    """Structure representing a training checkpoint."""

    epoch: int
    """Epoch number when the checkpoint was saved."""

    accuracy: float | None
    """Accuracy at the time of saving the checkpoint."""

    model_state_dict: dict[str, t.Any]
    """State dictionary of the model."""

    optimizer_state_dict: dict[str, t.Any] | None = None
    """State dictionary of the optimizer."""

    scheduler_state_dict: dict[str, t.Any] | None = None
    """State dictionary of the learning rate scheduler."""

    config: dict[str, t.Any] = {}
    """Configuration dictionary used for training."""

    classes: list[str] = []
    """List of class names."""

    class_to_idx: dict[str, int] = {}
    """Mapping from class names to indices."""

    extra_state: dict[str, t.Any] = {}
    """Any additional state information."""

    __vf_ckpt_version__: t.Final[int] = __vf_ckpt_version__
    """Checkpoint version."""

    def to_dict(self) -> dict[str, t.Any]:
        """Convert the Checkpoint to a dictionary."""
        return self._asdict()

    @classmethod
    def from_dict(cls, data: dict[str, t.Any]) -> Checkpoint:
        if "__vf_ckpt_version__" not in data:
            raise ValueError("Invalid checkpoint data: missing version info.")

        if (version := data["__vf_ckpt_version__"]) != 1:
            raise ValueError(f"Unsupported checkpoint version: {version}")

        return cls(**data)
