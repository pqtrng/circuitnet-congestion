"""Device selection: accelerator-then-CPU. No hardware or cloud vendor names."""

import torch


def get_device() -> torch.device:
    """Return the compute device.

    Prefers a CUDA-capable accelerator when available, otherwise falls back
    to CPU. Intentionally does not select MPS: canonical training and results
    are produced only on the accelerator/CPU path to keep numbers reproducible
    across machines.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
