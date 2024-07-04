"""
Custom implementation of TokenEditDistance metric to handle list of strings as input.
"""

from typing import Any, Literal, Union, Sequence
from torchmetrics.functional.text.helper import _LevenshteinEditDistance as _LE_distance
from torchmetrics.text import EditDistance
from torch import Tensor
import torch


def _edit_distance_update(
    preds: Union[str, Sequence[str]],
    target: Union[str, Sequence[str]],
    substitution_cost: int = 1,
) -> Tensor:
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]
    if not all(isinstance(x, str) for x in preds):
        raise ValueError(
            f"Expected all values in argument `preds` to be string type, but got {preds}"
        )
    if not all(isinstance(x, str) for x in target):
        raise ValueError(
            f"Expected all values in argument `target` to be string type, but got {target}"
        )
    if len(preds) != len(target):
        raise ValueError(
            f"Expected argument `preds` and `target` to have same length, but got {len(preds)} and {len(target)}"
        )

    distance = [
        _LE_distance(t.split(), op_substitute=substitution_cost)(p.split())[0] for p, t in zip(preds, target)  # type: ignore[arg-type]
    ]
    return torch.tensor(distance, dtype=torch.int)


class TokenEditDistance(EditDistance):
    """
    Computes the edit distance between two lists of tokens.
    """
    def __init__(
        self,
        substitution_cost: int = 1,
        reduction: None | Literal["mean"] | Literal["sum"] | Literal["none"] = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(substitution_cost, reduction, **kwargs)

    def update(self, preds: list[str], target: list[str]) -> None:
        """Update state with predictions and targets."""
        distance = _edit_distance_update(preds, target, self.substitution_cost)
        if self.reduction == "none" or self.reduction is None:
            self.edit_scores_list.append(distance)
        else:
            self.edit_scores += distance.sum()
            self.num_elements += distance.shape[0]
