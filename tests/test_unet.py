"""Tests for the congestion U-Net.

Runs on CPU with small synthetic inputs; continuous integration has no
accelerator. Spatial size is kept at 32x32, which is still divisible by the
default depth's downsampling factor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from circuitnet_congestion.models.unet import (
    DEFAULT_BASE_CHANNELS,
    DEFAULT_DEPTH,
    DEFAULT_IN_CHANNELS,
    UNet,
    count_parameters,
)

SIZE = 32
CANONICAL_RUNS = ("unet_a", "unet_b")


def test_output_shape_matches_input_resolution() -> None:
    model = UNet().eval()
    with torch.no_grad():
        output = model(torch.randn(2, DEFAULT_IN_CHANNELS, SIZE, SIZE))

    assert output.shape == (2, 1, SIZE, SIZE)


def test_default_parameter_count_is_pinned() -> None:
    """Guards the channel plan: a change here means the architecture moved."""
    assert count_parameters(UNet()) == 7_849_601


def test_count_parameters_respects_trainable_flag() -> None:
    model = UNet()
    total = count_parameters(model, trainable_only=False)
    for parameter in model.head.parameters():
        parameter.requires_grad_(False)

    assert count_parameters(model, trainable_only=True) < total


def test_head_is_zero_initialised() -> None:
    """See the head initialisation note in the module: the Kaiming rule computes
    a fan-out of one for this layer, and the marginal optimum of a sparse target
    is zero everywhere."""
    model = UNet()

    assert torch.count_nonzero(model.head.weight) == 0
    assert torch.count_nonzero(model.head.bias) == 0


def test_initial_prediction_is_exactly_zero() -> None:
    model = UNet().eval()
    with torch.no_grad():
        output = model(torch.randn(4, DEFAULT_IN_CHANNELS, SIZE, SIZE))

    assert torch.equal(output, torch.zeros_like(output))


def test_head_blocks_gradient_until_it_moves_off_zero() -> None:
    """A zero head receives gradient itself but transmits none backwards. Once
    it holds any non-zero weight the encoder starts learning."""
    model = UNet().train()
    inputs = torch.randn(2, DEFAULT_IN_CHANNELS, SIZE, SIZE)
    target = torch.rand(2, 1, SIZE, SIZE) * 0.1

    nn.functional.mse_loss(model(inputs), target).backward()
    first_encoder_conv = model.encoders[0].block[0]

    assert torch.count_nonzero(model.head.weight.grad) > 0
    assert torch.count_nonzero(first_encoder_conv.weight.grad) == 0

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.head.weight.normal_(std=0.01)
    nn.functional.mse_loss(model(inputs), target).backward()

    assert torch.count_nonzero(first_encoder_conv.weight.grad) > 0


def test_head_is_linear_and_admits_negative_output() -> None:
    """The output layer has no activation. Ground truth is non-negative, but a
    rectified head would place most predictions in its dead region given how
    many target pixels are zero."""
    model = UNet().eval()
    with torch.no_grad():
        model.head.weight.normal_(std=0.5)
        output = model(torch.randn(4, DEFAULT_IN_CHANNELS, SIZE, SIZE))

    assert float(output.min()) < 0.0


def test_convolutions_before_normalisation_carry_no_bias() -> None:
    for module in UNet().modules():
        if isinstance(module, nn.Sequential):
            for previous, following in zip(module[:-1], module[1:], strict=False):
                if isinstance(previous, nn.Conv2d) and isinstance(following, nn.BatchNorm2d):
                    assert previous.bias is None


def test_encoder_and_decoder_stages_are_paired() -> None:
    model = UNet(depth=3)

    assert len(model.encoders) == len(model.decoders) == 3
    assert model.size_divisor == 8


@pytest.mark.parametrize("depth", [1, 2, DEFAULT_DEPTH])
def test_varying_depth_preserves_resolution(depth: int) -> None:
    model = UNet(depth=depth, base_channels=4).eval()
    with torch.no_grad():
        output = model(torch.randn(1, DEFAULT_IN_CHANNELS, SIZE, SIZE))

    assert output.shape == (1, 1, SIZE, SIZE)


def test_custom_channel_configuration() -> None:
    model = UNet(in_channels=5, out_channels=2, base_channels=8, depth=2).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 5, SIZE, SIZE))

    assert output.shape == (1, 2, SIZE, SIZE)
    assert model.base_channels == DEFAULT_BASE_CHANNELS // 4


def test_rejects_non_four_dimensional_input() -> None:
    with pytest.raises(ValueError, match="expected a 4D input"):
        UNet()(torch.randn(DEFAULT_IN_CHANNELS, SIZE, SIZE))


def test_rejects_wrong_channel_count() -> None:
    with pytest.raises(ValueError, match="input channels"):
        UNet()(torch.randn(1, DEFAULT_IN_CHANNELS + 2, SIZE, SIZE))


def test_rejects_indivisible_spatial_size() -> None:
    with pytest.raises(ValueError, match="divisible by 16"):
        UNet()(torch.randn(1, DEFAULT_IN_CHANNELS, SIZE - 1, SIZE))


def test_rejects_invalid_construction_arguments() -> None:
    with pytest.raises(ValueError, match="depth must be at least 1"):
        UNet(depth=0)
    with pytest.raises(ValueError, match="base_channels must be at least 1"):
        UNet(base_channels=0)


def test_construction_is_deterministic_under_a_fixed_seed() -> None:
    torch.manual_seed(0)
    first = UNet(base_channels=4, depth=2)
    torch.manual_seed(0)
    second = UNet(base_channels=4, depth=2)

    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(left, right)


def test_default_parameter_count_is_pinned_and_matches_the_run_records() -> None:
    """Pin the size of the default architecture.

    The width choice is stated as a parameter count in the module docstring and
    in the README. Nothing enforced it: altering ``base_channels`` or ``depth``
    would leave both claims silently wrong, so a figure fixed by code drifted
    exactly like a measurement.

    The same number decides whether the checkpoints published for the completed
    runs still load into this code. A shape change makes every one of them
    unusable, and the failure would surface as a state-dict error during
    evaluation rather than here, where it belongs.
    """
    expected = 7_849_601
    assert count_parameters(UNet()) == expected

    root = Path(__file__).resolve().parents[1]
    for name in CANONICAL_RUNS:
        record = root / "results" / name / "run.json"
        payload = json.loads(record.read_text(encoding="utf-8"))
        assert payload["model"]["parameters"] == expected, name
