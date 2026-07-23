"""Smoke tests: import, version, device selection. No real data, CPU-safe."""

import torch

import circuitnet_congestion as pkg
from circuitnet_congestion.device import get_device


def test_package_imports():
    assert pkg is not None


def test_version_string():
    assert pkg.__version__ == "0.1.0"


def test_device_is_valid():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cuda", "cpu"}


def test_device_no_mps():
    # Guard the hardware-neutrality invariant: MPS must never be selected.
    assert get_device().type != "mps"
