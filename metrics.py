# Python STL
import logging
from typing import List
# PyTorch
import torch
import math
# Local
import utils
import torchsnooper

logger = logging.getLogger('')


# @torchsnooper.snoop()
def dice_score(preds: torch.Tensor,
               targets: torch.Tensor) -> torch.Tensor:
    batch_size: int = targets.shape[0]
    with torch.no_grad():
        # Shape: [N, H, W]
        p = preds.view(batch_size, -1).double()
        t = targets.view(batch_size, -1).double()
        # Shape: [N, H*W]
        if not (preds.shape == targets.shape):
            raise ValueError(f"Shape of preds: {preds.shape} must be the same"
                             f"as that of targets: {targets.shape}.")
        # Shape: [N, 1]
        dice = (2 * (p * t).sum(-1) + 1e-3) / ((p + t).sum(-1) + 1e-3)
        is_nan = torch.isnan(dice)
        dice[is_nan] = 0
        score = dice.sum() / (~is_nan).float().sum()
    return score


def true_positive(preds: torch.Tensor,
                  targets: torch.Tensor,
                  num_classes: int = 2) -> torch.Tensor:
    """Compute number of true positive predictions

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths
    num_classes : int
        Number of classes (including background)

    Returns
    -------
    tp : torch.Tensor
        Tensor of number of true positives for each class
    """
    out: List[torch.Tensor] = []
    for i in range(num_classes):
        out.append(((preds == i) & (targets == i)).sum())

    return torch.tensor(out)


def true_negative(preds: torch.Tensor,
                  targets: torch.Tensor,
                  num_classes: int) -> torch.Tensor:
    """Computes number of true negative predictions

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths
    num_classes : int
        Number of classes (including background)

    Returns
    -------
    tn : torch.Tensor
        Tensor of true negatives for each class
    """
    out: List[torch.Tensor] = []
    for i in range(num_classes):
        out.append(((preds != i) & (targets != i)).sum())

    return torch.tensor(out)


def false_positive(preds: torch.Tensor,
                   targets: torch.Tensor,
                   num_classes: int) -> torch.Tensor:
    """Computes number of false positive predictions

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths
    num_classes : int
        Number of classes (including background)

    Returns
    -------
    fp : torch.Tensor
        Tensor of false positives for each class
    """
    out: List[torch.Tensor] = []
    for i in range(num_classes):
        out.append(((preds == i) & (targets != i)).sum())

    return torch.tensor(out)


def false_negative(preds: torch.Tensor,
                   targets: torch.Tensor,
                   num_classes: int) -> torch.Tensor:
    """Computes number of false negative predictions

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths
    num_classes : int
        Number of classes (including background)

    Returns
    -------
    fn : torch.Tensor
        Tensor of false negatives for each class
    """
    out: List[torch.Tensor] = []
    for i in range(num_classes):
        out.append(((preds != i) & (targets == i)).sum())

    return torch.tensor(out)


def precision_score(preds: torch.Tensor,
                    targets: torch.Tensor,
                    num_classes: int = 2) -> torch.Tensor:
    """Computes precision score

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths
    num_classes : int
        Number of classes (including background)

    Returns
    -------
    precision : Tuple[torch.Tensor, ...]
        List of precision scores for each class
        return precision of class 1 (cells) not background
    """
    tp = true_positive(preds, targets, num_classes).to(torch.float)
    fp = false_positive(preds, targets, num_classes).to(torch.float)
    out = tp / (tp + fp)
    out[torch.isnan(out)] = 0

    return out[1]


def accuracy_score(preds: torch.Tensor,
                   targets: torch.Tensor,
                   smooth: float = 1e-10) -> torch.Tensor:
    """Compute accuracy score

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths
    smooth: float
        Smoothing for numerical stability
        1e-10 by default

    Returns
    -------
    acc : torch.Tensor
        Average accuracy score
    """
    valids = (targets >= 0)
    valids = valids.long()
    intersection = (preds == targets.long()).long()
    acc_sum = (valids * intersection).sum().float()
    valid_sum = valids.sum().float()
    return acc_sum / (valid_sum + smooth)


def iou_score(preds: torch.Tensor,
              targets: torch.Tensor,
              smooth: float = 1e-3) -> torch.Tensor:
    intersection = torch.sum(targets * preds)
    union = torch.sum(targets) + torch.sum(preds) - intersection + smooth
    score = (intersection + smooth) / union

    return score
