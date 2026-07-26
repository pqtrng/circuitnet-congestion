"""Tests for the shrinkage diagnostics.

The central case is the one that a loss curve cannot express: a model sitting at
the conditional-mean optimum of squared error, predicting a smooth field below
the hotspot threshold. Its error is genuinely low and it detects nothing.
"""

from __future__ import annotations

import pytest
import torch

from circuitnet_congestion.training.losses import DEFAULT_HOTSPOT_THRESHOLD
from circuitnet_congestion.training.metrics import (
    HotspotCounts,
    hotspot_counts,
    masked_max,
)

THRESHOLD = DEFAULT_HOTSPOT_THRESHOLD


def test_masked_max_ignores_padded_pixels() -> None:
    values = torch.zeros(1, 1, 4, 4)
    mask = torch.ones(1, 1, 4, 4)
    values[0, 0, 0, 0] = 0.3
    mask[0, 0, 2:, :] = 0.0
    values[0, 0, 3, 3] = 99.0

    assert masked_max(values, mask).item() == pytest.approx(0.3)


def test_masked_max_handles_negative_values() -> None:
    """The output head is linear, so early predictions can be entirely negative."""
    values = torch.full((1, 1, 3, 3), -2.0)
    values[0, 0, 1, 1] = -0.5

    assert masked_max(values, torch.ones(1, 1, 3, 3)).item() == pytest.approx(-0.5)


def test_masked_max_of_an_empty_mask_is_zero() -> None:
    values = torch.randn(1, 1, 4, 4)

    assert masked_max(values, torch.zeros(1, 1, 4, 4)).item() == 0.0


def test_shrunk_model_raises_every_alarm() -> None:
    """A constant prediction at the conditional mean of a hotspot with
    probability 0.3 and value 0.07. This minimises squared error and detects
    nothing; the loss curve alone would call it healthy."""
    target = torch.zeros(1, 1, 100, 1)
    target[0, 0, :30, 0] = 0.07
    mask = torch.ones(1, 1, 100, 1)
    prediction = torch.full_like(target, 0.021)

    counts = hotspot_counts(prediction, target, mask, THRESHOLD)

    assert masked_max(prediction, mask).item() < THRESHOLD
    assert counts.predicted_positive == 0
    assert counts.false_negative == 30
    assert counts.recall == 0.0
    assert counts.precision == 0.0
    assert counts.f1 == 0.0


def test_perfect_prediction_scores_one() -> None:
    target = torch.zeros(1, 1, 100, 1)
    target[0, 0, :30, 0] = 0.07
    mask = torch.ones(1, 1, 100, 1)

    counts = hotspot_counts(target.clone(), target, mask, THRESHOLD)

    assert counts.true_positive == 30
    assert counts.recall == 1.0
    assert counts.precision == 1.0
    assert counts.f1 == 1.0


def test_counts_are_restricted_to_valid_pixels() -> None:
    prediction = torch.tensor([[[[0.9, 0.9, 0.0, 0.0]]]])
    target = torch.tensor([[[[0.9, 0.0, 0.9, 0.0]]]])
    mask = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])

    counts = hotspot_counts(prediction, target, mask, THRESHOLD)

    assert counts == HotspotCounts(true_positive=1, false_positive=0, false_negative=1)


def test_threshold_comparison_is_strict() -> None:
    """Targets land on a grid of small-denominator fractions, so a value
    exactly at the threshold must fall on one side deterministically."""
    at_threshold = torch.full((1, 1, 1, 1), THRESHOLD)
    mask = torch.ones(1, 1, 1, 1)

    counts = hotspot_counts(at_threshold, at_threshold, mask, THRESHOLD)

    assert counts == HotspotCounts()


def test_counts_are_additive_across_batches() -> None:
    """An epoch total is the sum of its batches; rates averaged per batch are
    not the rate over the epoch when batches hold different pixel counts."""
    torch.manual_seed(0)
    prediction = torch.rand(6, 1, 8, 8) * 0.12
    target = (torch.rand(6, 1, 8, 8) < 0.2).float() * 0.09
    mask = (torch.rand(6, 1, 8, 8) > 0.3).float()

    whole = hotspot_counts(prediction, target, mask, THRESHOLD)
    parts = sum(
        (
            hotspot_counts(prediction[i : i + 1], target[i : i + 1], mask[i : i + 1], THRESHOLD)
            for i in range(6)
        ),
        HotspotCounts(),
    )

    assert whole == parts


def test_pixelwise_rates_differ_from_batch_averaged_rates() -> None:
    """Concretely: one batch with many hotspots, one with a single pixel."""
    dense_prediction = torch.full((1, 1, 10, 10), 0.9)
    dense_target = torch.full((1, 1, 10, 10), 0.9)
    sparse_prediction = torch.zeros(1, 1, 10, 10)
    sparse_target = torch.full((1, 1, 10, 10), 0.9)
    mask = torch.ones(1, 1, 10, 10)
    sparse_mask = torch.zeros(1, 1, 10, 10)
    sparse_mask[0, 0, 0, 0] = 1.0

    total = hotspot_counts(dense_prediction, dense_target, mask, THRESHOLD) + hotspot_counts(
        sparse_prediction, sparse_target, sparse_mask, THRESHOLD
    )
    averaged_recall = (1.0 + 0.0) / 2

    assert total.recall == pytest.approx(100 / 101)
    assert total.recall != pytest.approx(averaged_recall)


def test_derived_properties() -> None:
    counts = HotspotCounts(true_positive=6, false_positive=2, false_negative=3)

    assert counts.predicted_positive == 8
    assert counts.target_positive == 9
    assert counts.precision == pytest.approx(0.75)
    assert counts.recall == pytest.approx(6 / 9)
    assert counts.f1 == pytest.approx(12 / 17)


def test_empty_counts_report_zero_rather_than_dividing_by_zero() -> None:
    empty = HotspotCounts()

    assert empty.precision == 0.0
    assert empty.recall == 0.0
    assert empty.f1 == 0.0


def test_counts_are_immutable() -> None:
    counts = HotspotCounts()

    with pytest.raises(AttributeError):
        counts.true_positive = 1  # type: ignore[misc]


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        hotspot_counts(
            torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 8, 8), torch.ones(1, 1, 4, 4), THRESHOLD
        )
    with pytest.raises(ValueError, match="shape mismatch"):
        hotspot_counts(
            torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 8, 8), THRESHOLD
        )
