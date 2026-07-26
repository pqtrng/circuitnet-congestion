"""Masked losses and reduction utilities for sparse congestion regression.

Reduction. Every quantity here is reduced as a single global masked mean over
all valid pixels in the batch -- sum(value * mask) / sum(mask) -- rather than as
a mean of per-patch means. Patch coverage ranges from 1.6% to 100% of the
128x128 window, so averaging per-patch means would make each pixel of a small
edge fragment count up to sixty-four times as much as a pixel of a full patch.
The unit of this prediction task is a grid cell on the die, not a patch. Edge
fragments are kept rather than filtered: the die boundary is where macros and IO
sit, so discarding it would discard the hard cases.

A consequence: the loss depends mildly on batch composition, and any gradient
accumulation must sum numerators and denominators separately instead of
averaging batch losses. MaskedMeanAccumulator exists for exactly that.

Target distribution, measured on the train split over valid pixels only:

    non-zero fraction              1.52%
    mean of squared targets        3.31e-05   (the loss of a zero predictor)
    targets above 0.05             0.13% of pixels, 67% of total squared error
    targets above 0.20             0.0014% of pixels, 40% of total squared error

Targets are quantised: they are routing overflow divided by track capacity, so
they take fractional values with small denominators that vary by design and
metal layer (1/44, 1/34, 2/43, 3/44, ...). The hotspot threshold below is placed
between two adjacent grid levels for that reason.

Two objectives are provided because plain squared error fails on this
distribution in a specific and measurable way. Squaring already amplifies large
targets, so the failure is not that rare high values are ignored. The failure is
shrinkage: the minimiser of squared error at an uncertain pixel is the
conditional mean, which for a pixel that is a hotspot with probability 0.3 and
value 0.07 sits at 0.021 -- below any usable hotspot threshold. The model then
predicts a smooth low-amplitude field that scores well per pixel while crossing
no threshold anywhere. The weighted variant applies a binary class weight, not a
magnitude weight, and its default was derived by solving the same conditional
optimum for the weight at which it clears the threshold.
"""

from __future__ import annotations

import torch
from torch import nn

DEFAULT_HOTSPOT_THRESHOLD = 0.05
DEFAULT_POSITIVE_WEIGHT = 10.0

LOSS_MSE = "masked_mse"
LOSS_WEIGHTED_MSE = "masked_weighted_mse"


def _validate_shapes(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} does not match "
            f"target shape {tuple(target.shape)}"
        )
    if prediction.shape != mask.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} does not match "
            f"mask shape {tuple(mask.shape)}"
        )


def masked_sum_and_weight(
    values: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the masked weighted sum of ``values`` and its total weight.

    Returning the two terms separately is what allows the same global mean to be
    accumulated correctly across batches of differing valid-pixel counts.
    """
    effective = mask if weight is None else mask * weight
    return (values * effective).sum(), effective.sum()


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Global masked mean over valid pixels. Returns zero if none are valid."""
    total, count = masked_sum_and_weight(values, mask, weight)
    return total / count.clamp_min(torch.finfo(total.dtype).tiny)


def hotspot_weight(
    target: torch.Tensor,
    *,
    threshold: float = DEFAULT_HOTSPOT_THRESHOLD,
    positive_weight: float = DEFAULT_POSITIVE_WEIGHT,
) -> torch.Tensor:
    """Binary class weight: ``positive_weight`` above the threshold, one below.

    The weight depends only on the target, never on the prediction, so the
    normalising denominator it induces is constant with respect to the model
    parameters and does not distort the gradient.
    """
    return torch.where(
        target > threshold,
        torch.full_like(target, positive_weight),
        torch.ones_like(target),
    )


def zero_predictor_mse(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean squared error attained by predicting zero everywhere.

    This is the trivial baseline of the task and equals the masked mean of the
    squared targets. On a target where 98.5% of valid pixels are zero, a small
    absolute error is not by itself evidence of anything, so no reported error
    figure is interpretable without this number beside it.
    """
    return masked_mean(target.square(), mask)


class MaskedMSELoss(nn.Module):
    """Squared error over valid pixels, reduced as a single global mean."""

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        _validate_shapes(prediction, target, mask)
        return masked_mean((prediction - target).square(), mask)

    def extra_repr(self) -> str:
        return "reduction=global_masked_mean"


class MaskedWeightedMSELoss(nn.Module):
    """Squared error with a binary class weight on hotspot pixels.

    The weight is binary rather than proportional to the target: squaring
    already amplifies large targets, and 0.13% of valid pixels carry 67% of the
    unweighted squared error, so a magnitude-proportional weight would compound
    an existing bias instead of correcting one.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_HOTSPOT_THRESHOLD,
        positive_weight: float = DEFAULT_POSITIVE_WEIGHT,
    ) -> None:
        super().__init__()
        if positive_weight <= 0.0:
            raise ValueError(f"positive_weight must be positive, got {positive_weight}")
        self.threshold = threshold
        self.positive_weight = positive_weight

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        _validate_shapes(prediction, target, mask)
        weight = hotspot_weight(
            target,
            threshold=self.threshold,
            positive_weight=self.positive_weight,
        )
        return masked_mean((prediction - target).square(), mask, weight)

    def extra_repr(self) -> str:
        return (
            f"threshold={self.threshold}, positive_weight={self.positive_weight}, "
            f"reduction=global_masked_mean"
        )


class MaskedMeanAccumulator:
    """Accumulates a global masked mean across batches.

    Averaging per-batch losses would silently reweight batches by their valid
    pixel count, and the final batch of an epoch is usually short. This keeps
    the numerator and the denominator apart until the end.
    """

    def __init__(self) -> None:
        self.total = 0.0
        self.weight = 0.0

    def update(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            total, count = masked_sum_and_weight(values.detach(), mask, weight)
        self.total += float(total)
        self.weight += float(count)

    def compute(self) -> float:
        if self.weight == 0.0:
            return 0.0
        return self.total / self.weight

    def reset(self) -> None:
        self.total = 0.0
        self.weight = 0.0


def build_loss(name: str, **kwargs: float) -> nn.Module:
    """Instantiate a loss by configuration name."""
    if name == LOSS_MSE:
        if kwargs:
            raise ValueError(f"{LOSS_MSE} takes no parameters, got {sorted(kwargs)}")
        return MaskedMSELoss()
    if name == LOSS_WEIGHTED_MSE:
        return MaskedWeightedMSELoss(**kwargs)
    raise ValueError(f"unknown loss {name!r}; expected one of {(LOSS_MSE, LOSS_WEIGHTED_MSE)}")
