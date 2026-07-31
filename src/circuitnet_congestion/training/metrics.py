"""Diagnostics for detecting shrinkage during training.

Squared error on this target admits a failure mode that its own curve cannot
reveal. Where the model cannot distinguish pixels, the minimiser is the
conditional mean, so the model settles on a smooth low-amplitude field: the
error falls epoch after epoch while no pixel anywhere crosses the hotspot
threshold. A run in that state looks healthy on a loss plot and is useless.

The quantities here exist to make that state visible within a few epochs rather
than after a full training budget. They are deliberately narrow -- the full
evaluation suite belongs with evaluation, not with the training loop.

Counts are returned rather than rates. A rate computed per batch and averaged
across batches is not the rate over the epoch unless every batch holds the same
number of valid pixels, which is never the case here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Largest value over valid pixels; zero when no pixel is valid.

    Tracked every epoch as the cheapest possible shrinkage alarm: a prediction
    whose maximum stays below the hotspot threshold has collapsed to a smooth
    field regardless of what the loss curve shows.

    The empty-mask return is zero, which is inside the range of legitimate
    results rather than outside it -- unlike the best-F1 sentinel, which is
    negative precisely so that no real score can be mistaken for it. Zero is
    tolerable here only because of which way it errs: it reads as a collapsed
    prediction and trips the alarm rather than silencing it. It is not
    reachable with the gold layer, whose least-covered patch still has valid
    pixels; a batch of pure padding would mean the loader is wrong.
    """
    neutral = torch.full_like(values, torch.finfo(values.dtype).min)
    largest = torch.where(mask > 0, values, neutral).max()
    return torch.where(mask.sum() > 0, largest, torch.zeros_like(largest))


@dataclass(frozen=True)
class HotspotCounts:
    """Confusion counts for thresholded hotspot detection over valid pixels.

    Instances are additive so that an epoch total is the sum of its batches.
    """

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def __add__(self, other: HotspotCounts) -> HotspotCounts:
        return HotspotCounts(
            true_positive=self.true_positive + other.true_positive,
            false_positive=self.false_positive + other.false_positive,
            false_negative=self.false_negative + other.false_negative,
        )

    @property
    def target_positive(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def predicted_positive(self) -> int:
        return self.true_positive + self.false_positive

    @property
    def precision(self) -> float:
        """Zero when nothing was predicted positive, which is the state a
        shrunk model is in; the value is reported rather than left undefined."""
        if self.predicted_positive == 0:
            return 0.0
        return self.true_positive / self.predicted_positive

    @property
    def recall(self) -> float:
        if self.target_positive == 0:
            return 0.0
        return self.true_positive / self.target_positive

    @property
    def f1(self) -> float:
        denominator = self.predicted_positive + self.target_positive
        if denominator == 0:
            return 0.0
        return 2 * self.true_positive / denominator


def hotspot_counts(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
) -> HotspotCounts:
    """Confusion counts at a fixed threshold, restricted to valid pixels."""
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError(
            f"shape mismatch: prediction {tuple(prediction.shape)}, "
            f"target {tuple(target.shape)}, mask {tuple(mask.shape)}"
        )

    valid = mask > 0
    predicted_positive = (prediction > threshold) & valid
    target_positive = (target > threshold) & valid

    return HotspotCounts(
        true_positive=int((predicted_positive & target_positive).sum()),
        false_positive=int((predicted_positive & ~target_positive).sum()),
        false_negative=int((~predicted_positive & target_positive).sum()),
    )
