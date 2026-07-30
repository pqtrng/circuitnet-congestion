"""Masked losses and reduction utilities for sparse congestion regression.

Reduction. Every quantity here is reduced as a single global masked mean over
all valid pixels in the batch -- sum(value * mask) / sum(mask) -- rather than as
a mean of per-patch means. Patch coverage varies widely, because edge patches
are zero-padded to the full window, so averaging per-patch means would let a
pixel in a small edge fragment outweigh a pixel in a full patch by the inverse
of the smallest coverage. The unit of this prediction task is a grid cell on the
die, not a patch. Edge fragments are kept rather than filtered: the die boundary
is where macros and IO sit, so discarding it would discard the hard cases.

A consequence: the loss depends mildly on batch composition, and any gradient
accumulation must sum numerators and denominators separately instead of
averaging batch losses. MaskedMeanAccumulator exists for exactly that.

Target distribution. The training targets are overwhelmingly zero, and the
squared error of the non-zero remainder is concentrated in a small tail. Both
facts are measured rather than assumed, and the figures are not restated here:
they live in ``results/probes/target_stats.json`` under ``splits.train``, as
``nonzero_fraction_of_valid``, ``zero_predictor_mse`` and ``error_share_above``.
A measurement copied into a docstring drifts away from the artefact it came
from, and the ones that used to sit in this paragraph had already done so.

Targets are quantised: they are routing overflow divided by track capacity, so
they take fractional values with small denominators that vary by design and
metal layer. The training split holds many distinct non-zero levels
(``splits.train.distinct_nonzero_values``), and around the configured threshold
they are dense enough that no threshold sits in a meaningful gap between
neighbours (``splits.train.levels_near_threshold``).

The threshold is therefore defined physically rather than geometrically: a
hotspot is a cell whose routing demand exceeds its track capacity by more than
DEFAULT_HOTSPOT_THRESHOLD. That value is itself one of the quantisation levels
present in the data rather than a gap between two of them, and the strict
comparison below excludes the pixels sitting exactly on it. Their count is
recorded per split as ``pixels_on_threshold``; measured against the pixels the
strict comparison does admit, the excluded group is not negligible and its
relative size differs between splits, which is why evaluation reports a range of
thresholds rather than a single number.

Determinism of that boundary depends on the comparison happening in the tensors'
own dtype. Targets are stored as float32; the Python threshold constant is
weak-promoted to float32 before the comparison, where it coincides exactly with
the stored level, so the level is excluded. Widened to float64 the two no longer
coincide, the level is admitted instead, and every pixel sitting on it changes
class. Any code that upcasts before thresholding is making a different decision
than this loss makes, and offline analysis that works in float64 has to round
back to agree with it.

Two objectives are provided because plain squared error fails on this
distribution in a specific and measurable way. Squaring already amplifies large
targets, so the failure is not that rare high values are ignored. The failure is
shrinkage: the minimiser of squared error at an uncertain pixel is the
conditional mean, and for a pixel that is a hotspot with probability p and value
y that optimum sits at p*y, below the threshold whenever the pixel is genuinely
uncertain. The model then predicts a smooth low-amplitude field that scores well
per pixel while crossing no threshold anywhere.

The weighted variant applies a binary class weight rather than a magnitude
weight. Under weight w on hotspot pixels the same optimum becomes
w*p*y / (w*p + 1 - p), which increases with w, so a large enough w lifts it back
over the threshold. Solving that inequality gives a lower bound on w -- but the
bound is a function of p and y, and neither is measured: any pair of values for
them is a hypothesis about how uncertain one particular pixel is.
DEFAULT_POSITIVE_WEIGHT is a round number placed above the bound that a
plausible worked case implies, with margin. The derivation constrains it from
below; the margin is a choice, and the run records are what decide whether the
choice was a good one.
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
    squared targets. Almost every valid pixel is zero, so a small absolute error
    is not by itself evidence of anything, and no reported error figure is
    interpretable without this number beside it. The zero fraction differs by
    split; both it and this baseline are recorded per split in
    ``results/probes/target_stats.json``.
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
    already amplifies large targets, and on the training split a small fraction
    of valid pixels carries most of the unweighted squared error -- see
    ``splits.train.error_share_above`` in ``results/probes/target_stats.json`` --
    so a magnitude-proportional weight would compound an existing bias instead
    of correcting one.
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
