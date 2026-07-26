"""Tests for masked losses.

These pin down three things that are easy to break silently: that padded pixels
cannot influence any reported number, that reduction is a single global mean
rather than a mean of per-patch means, and that the default hotspot weight is
the value derived from the shrinkage argument rather than a number someone liked.
"""

from __future__ import annotations

import pytest
import torch

from circuitnet_congestion.training.losses import (
    DEFAULT_HOTSPOT_THRESHOLD,
    DEFAULT_POSITIVE_WEIGHT,
    LOSS_MSE,
    LOSS_WEIGHTED_MSE,
    MaskedMeanAccumulator,
    MaskedMSELoss,
    MaskedWeightedMSELoss,
    build_loss,
    hotspot_weight,
    masked_mean,
    masked_sum_and_weight,
    zero_predictor_mse,
)


def test_padded_pixels_cannot_influence_the_loss() -> None:
    prediction = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    mask = torch.ones(1, 1, 4, 4)
    target[0, 0, 0, 0] = 0.08
    mask[0, 0, 2:, :] = 0.0
    target[0, 0, 3, 3] = 99.0

    loss = MaskedMSELoss()(prediction, target, mask)

    assert loss.item() == pytest.approx(0.08**2 / 8)


def test_reduction_is_global_not_a_mean_of_patch_means() -> None:
    """A one-pixel edge fragment must not carry the weight of a full patch."""
    prediction = torch.zeros(2, 1, 8, 8)
    target = torch.zeros(2, 1, 8, 8)
    target[0] = 0.1
    mask = torch.zeros(2, 1, 8, 8)
    mask[0] = 1.0
    mask[1, 0, 0, 0] = 1.0

    loss_fn = MaskedMSELoss()
    combined = loss_fn(prediction, target, mask)
    per_patch = torch.stack(
        [loss_fn(prediction[i : i + 1], target[i : i + 1], mask[i : i + 1]) for i in range(2)]
    ).mean()

    assert combined.item() == pytest.approx(64 * 0.01 / 65)
    assert per_patch.item() == pytest.approx(0.005)
    assert not torch.allclose(combined, per_patch)


def test_zero_predictor_baseline_equals_mean_of_squared_targets() -> None:
    """The trivial baseline of the task: no error figure is interpretable
    without it, given that ~98.5% of valid target pixels are zero."""
    torch.manual_seed(0)
    target = (torch.rand(3, 1, 16, 16) < 0.02).float() * 0.07
    mask = (torch.rand(3, 1, 16, 16) > 0.25).float()

    baseline = zero_predictor_mse(target, mask)

    assert baseline.item() == pytest.approx(
        MaskedMSELoss()(torch.zeros_like(target), target, mask).item()
    )
    assert baseline.item() == pytest.approx(masked_mean(target.square(), mask).item())


def test_masked_sum_and_weight_returns_unreduced_terms() -> None:
    values = torch.full((1, 1, 2, 2), 3.0)
    mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])

    total, weight = masked_sum_and_weight(values, mask)

    assert total.item() == pytest.approx(6.0)
    assert weight.item() == pytest.approx(2.0)


def test_hotspot_weight_is_binary_and_target_driven() -> None:
    target = torch.tensor([[[[0.0, 0.04, 0.05, 0.06]]]])

    weight = hotspot_weight(target)

    assert weight.flatten().tolist() == [1.0, 1.0, 1.0, DEFAULT_POSITIVE_WEIGHT]


def test_hotspot_threshold_sits_between_two_quantisation_levels() -> None:
    """Targets are routing overflow over track capacity, so they land on a grid
    of small-denominator fractions. The threshold must not coincide with one."""
    assert 2 / 44 < DEFAULT_HOTSPOT_THRESHOLD < 3 / 44


def test_weighted_loss_reduces_to_plain_mse_at_unit_weight() -> None:
    torch.manual_seed(0)
    prediction = torch.randn(2, 1, 8, 8) * 0.05
    target = (torch.rand(2, 1, 8, 8) < 0.1).float() * 0.1
    mask = torch.ones(2, 1, 8, 8)

    weighted = MaskedWeightedMSELoss(positive_weight=1.0)(prediction, target, mask)
    plain = MaskedMSELoss()(prediction, target, mask)

    assert weighted.item() == pytest.approx(plain.item())


@pytest.mark.parametrize("weight", [1.0, 2.0, 5.0, 10.0, 15.0])
def test_shared_parameter_optimum_matches_the_closed_form(weight: float) -> None:
    """The failure mode of squared error here is shrinkage, not omission: where
    the model cannot distinguish pixels, the minimiser is the conditional mean.
    The closed form below is what sets the default positive weight."""
    probability, value, count = 0.3, 0.07, 1000
    positives = int(probability * count)
    target = torch.zeros(1, 1, count, 1)
    target[0, 0, :positives, 0] = value
    mask = torch.ones(1, 1, count, 1)

    expected = weight * probability * value / (weight * probability + (1 - probability))
    theta = torch.tensor([expected], requires_grad=True)
    MaskedWeightedMSELoss(positive_weight=weight)(
        theta.expand(count).reshape(1, 1, count, 1), target, mask
    ).backward()

    assert theta.grad is not None
    assert float(theta.grad.abs()) < 1e-7


def test_default_weight_lifts_the_shrunk_optimum_over_the_threshold() -> None:
    probability, value = 0.3, 0.07
    unweighted = probability * value
    weighted = (
        DEFAULT_POSITIVE_WEIGHT
        * probability
        * value
        / (DEFAULT_POSITIVE_WEIGHT * probability + (1 - probability))
    )

    assert unweighted < DEFAULT_HOTSPOT_THRESHOLD
    assert weighted > DEFAULT_HOTSPOT_THRESHOLD


def test_accumulator_matches_a_single_pass() -> None:
    torch.manual_seed(0)
    values = torch.randn(5, 1, 16, 16)
    mask = (torch.rand(5, 1, 16, 16) > 0.3).float()

    accumulator = MaskedMeanAccumulator()
    for index in range(5):
        accumulator.update(values[index : index + 1], mask[index : index + 1])

    assert accumulator.compute() == pytest.approx(masked_mean(values, mask).item(), rel=1e-6)


def test_accumulator_weights_batches_by_valid_pixels_not_by_batch() -> None:
    """The last batch of an epoch is usually short; averaging per-batch losses
    would give its pixels more weight than the rest."""
    large_values = torch.full((1, 1, 10, 10), 1.0)
    large_mask = torch.ones(1, 1, 10, 10)
    small_values = torch.full((1, 1, 10, 10), 0.0)
    small_mask = torch.zeros(1, 1, 10, 10)
    small_mask[0, 0, 0, 0] = 1.0

    accumulator = MaskedMeanAccumulator()
    accumulator.update(large_values, large_mask)
    accumulator.update(small_values, small_mask)

    assert accumulator.compute() == pytest.approx(100 / 101)


def test_accumulator_reset_and_empty_state() -> None:
    accumulator = MaskedMeanAccumulator()
    assert accumulator.compute() == 0.0

    accumulator.update(torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2))
    assert accumulator.compute() == pytest.approx(1.0)

    accumulator.reset()
    assert accumulator.compute() == 0.0


def test_accumulator_does_not_retain_the_graph() -> None:
    values = torch.ones(1, 1, 2, 2, requires_grad=True)
    accumulator = MaskedMeanAccumulator()

    accumulator.update(values, torch.ones(1, 1, 2, 2))

    assert isinstance(accumulator.total, float)


def test_fully_masked_input_yields_a_finite_zero() -> None:
    torch.manual_seed(0)
    loss = MaskedMSELoss()(
        torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)
    )

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_gradient_is_finite_and_flows_to_the_prediction() -> None:
    prediction = torch.zeros(2, 1, 8, 8, requires_grad=True)
    target = (torch.rand(2, 1, 8, 8) < 0.1).float() * 0.1
    mask = (torch.rand(2, 1, 8, 8) > 0.2).float()

    MaskedWeightedMSELoss()(prediction, target, mask).backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) > 0


@pytest.mark.parametrize("loss_fn", [MaskedMSELoss(), MaskedWeightedMSELoss()])
def test_shape_mismatch_is_rejected(loss_fn: torch.nn.Module) -> None:
    prediction = torch.zeros(1, 1, 4, 4)

    with pytest.raises(ValueError, match="target shape"):
        loss_fn(prediction, torch.zeros(1, 1, 8, 8), torch.ones(1, 1, 4, 4))
    with pytest.raises(ValueError, match="mask shape"):
        loss_fn(prediction, torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 8, 8))


def test_rejects_non_positive_weight() -> None:
    with pytest.raises(ValueError, match="positive_weight must be positive"):
        MaskedWeightedMSELoss(positive_weight=0.0)


def test_build_loss_dispatches_by_name() -> None:
    assert isinstance(build_loss(LOSS_MSE), MaskedMSELoss)

    weighted = build_loss(LOSS_WEIGHTED_MSE, positive_weight=3.0, threshold=0.02)
    assert isinstance(weighted, MaskedWeightedMSELoss)
    assert weighted.positive_weight == 3.0
    assert weighted.threshold == 0.02


def test_build_loss_rejects_unknown_names_and_stray_parameters() -> None:
    with pytest.raises(ValueError, match="unknown loss"):
        build_loss("huber")
    with pytest.raises(ValueError, match="takes no parameters"):
        build_loss(LOSS_MSE, positive_weight=2.0)


def test_loss_configuration_appears_in_repr() -> None:
    """Run logs record the loss object; its parameters have to be visible there."""
    text = repr(MaskedWeightedMSELoss(threshold=0.05, positive_weight=10.0))

    assert "threshold=0.05" in text
    assert "positive_weight=10.0" in text
    assert "global_masked_mean" in text
