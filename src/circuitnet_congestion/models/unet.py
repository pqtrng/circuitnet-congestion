"""U-Net for dense routing-congestion regression.

Design decisions worth stating explicitly, because each one is load-bearing for
the evaluation argument this project is built around:

Skip connections. Congestion hotspots are a sparse, high-frequency signal --
roughly 1.5% of valid pixels are non-zero. Pooling destroys exactly that band.
Without skips the decoder produces smooth blurred fields, which score well on
pixel-averaged metrics while missing every hotspot. The architecture therefore
has to be able to represent sharp localised structure before any claim about the
gap between pixel metrics and hotspot metrics can be made honestly.

Bilinear upsampling rather than transposed convolution. Transposed convolutions
generate periodic checkerboard artefacts. On a sparse hotspot map those artefacts
are indistinguishable from predicted hotspots and would enter the evaluation as
structured false positives.

Linear output. Ground truth is non-negative, which makes a ReLU head tempting.
It is a trap here: with the overwhelming majority of target pixels at zero, a
ReLU head parks most outputs in its dead region, where the gradient is exactly
zero for the bulk of the data. The head is linear; small negative predictions are
accepted during training and clamped at evaluation time.

Convolutions preceding a normalisation layer carry no bias term -- the batch-norm
shift parameter already provides it.

Mixed precision is available but off by default. Measured peak allocation for
this configuration is well under the available budget, so half precision buys
no memory that is needed; meanwhile targets occupy a range around 1e-2 and
squared errors fall to 1e-6, which sits at the bottom of half precision's
dynamic range. The accelerators this baseline is developed on also execute
half precision at the same rate as single precision, so there is no throughput
argument either.
"""

from __future__ import annotations

import torch
from torch import nn

DEFAULT_IN_CHANNELS = 3
DEFAULT_BASE_CHANNELS = 32
DEFAULT_DEPTH = 4


class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by normalisation and a ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Encoder-decoder with skip connections for image-to-image regression.

    At the default depth of 4 the spatial resolution follows
    128 -> 64 -> 32 -> 16 -> 8, so a bottleneck unit summarises a 16x16 input
    region and the theoretical receptive field spans most of the patch. Routing
    demand is a competition between regions, not a local property, so that reach
    is required rather than incidental.

    Default width is deliberately modest (base 32, ~7.8M parameters). The
    constraint is not capacity but iteration count: this baseline is trained
    twice under two different loss formulations, and it has to fit alongside
    mixed-precision activations at a batch size large enough to keep batch
    normalisation statistics stable.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = 1,
        base_channels: int = DEFAULT_BASE_CHANNELS,
        depth: int = DEFAULT_DEPTH,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        if base_channels < 1:
            raise ValueError(f"base_channels must be at least 1, got {base_channels}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth

        stage_channels = [base_channels * 2**level for level in range(depth)]
        bottleneck_channels = base_channels * 2**depth

        self.encoders = nn.ModuleList()
        previous = in_channels
        for channels in stage_channels:
            self.encoders.append(DoubleConv(previous, channels))
            previous = channels

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.bottleneck = DoubleConv(previous, bottleneck_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.decoders = nn.ModuleList()
        previous = bottleneck_channels
        for channels in reversed(stage_channels):
            # Input is the upsampled feature map concatenated with its skip.
            self.decoders.append(DoubleConv(previous + channels, channels))
            previous = channels

        self.head = nn.Conv2d(previous, out_channels, kernel_size=1)

        self._initialise_weights()

    @property
    def size_divisor(self) -> int:
        """Spatial dimensions must be divisible by this for skips to align."""
        return 2**self.depth

    def _initialise_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # The 1x1 head is initialised to zero rather than by the Kaiming rule
        # above. Two reasons. First, correctness: Kaiming with mode="fan_out" on
        # a 1x1 convolution having a single output channel computes fan_out = 1,
        # hence a weight standard deviation of sqrt(2), which yields predictions
        # two orders of magnitude larger than any target in this dataset. Second,
        # prior: roughly 98.5% of valid target pixels are zero, so a model that
        # begins by predicting zero everywhere begins near the marginal optimum
        # and descends from a well-conditioned point.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected a 4D input [B, C, H, W], got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {x.shape[1]}")

        divisor = self.size_divisor
        height, width = x.shape[-2:]
        if height % divisor or width % divisor:
            raise ValueError(
                f"spatial dimensions {(height, width)} must be divisible by {divisor} "
                f"for depth {self.depth}"
            )

        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips), strict=True):
            x = self.upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return self.head(x)


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    """Total parameter count, reported in run metadata for reproducibility."""
    parameters = model.parameters()
    if trainable_only:
        parameters = (p for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in parameters)
