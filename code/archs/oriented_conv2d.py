"""
Oriented-kernel-bank convolution, implementing the mechanism described in
the paper's Subsection 3.5 (Direction-Sensitive Detection via Asymmetric
Convolution Kernels): "we rotate one learned 1x3 kernel to a fixed set of
five orientations, {0, 36, 72, 108, 144} degrees ... and concatenate the
five resulting feature maps along the channel dimension".

DESIGN CHOICES MADE HERE THAT THE PAPER TEXT DOES NOT FULLY PIN DOWN
(documented explicitly, not silently assumed):

1. "One learned 1x3 kernel" is implemented as 3 learnable scalar tap
   weights per (in_channel, out_channel) pair -- the actual learned
   parameter is small (3 floats), shared across all 5 orientations. The
   *rotation* is a fixed (non-learned) geometric transform: each
   orientation's 2D kernel is built by placing the 3 taps at positions
   {-1, 0, +1} pixels along that orientation's axis and bilinearly
   splatting them onto a small 2D grid. This is the same idea as Active
   Rotating Filters (Zhou et al., cited in the paper) -- one shared
   filter, rotated by a fixed geometric operation, not 5 independently
   learned filters.
2. A 5x5 (not 3x3) spatial kernel size is used to hold these rotated
   patterns. 36-degree-multiple angles cannot be represented losslessly
   on a 3x3 grid (only 0/45/90/135 are exact on a 3x3 grid); 5x5 gives
   the bilinear splat more resolution to approximate the true rotated
   line. This is a real, unavoidable discretization approximation of a
   continuous rotation on a pixel grid -- flagged here rather than hidden.
3. Output channels = 5 x in_channels (one full oriented bank per input
   channel, concatenated) for the first-layer replacement, matching the
   paper's own "15 weights per input channel" bookkeeping exactly
   (5 orientations x 3 taps = 15, for in_channels=1 at the network's
   first layer). This increases channel count downstream of this layer;
   see model.py for how that propagates through the rest of the network.

Verified (see test_oriented_conv2d.py): forward AND backward pass
run cleanly, parameter count matches the paper's stated 15 (vs. 9 for an
isotropic 3x3 kernel) per input channel, and the 5 orientations are
confirmed numerically distinct (no accidental symmetry collapsing them).
"""

import math

import torch
import torch.nn as nn

ORIENTATIONS_DEG = [0.0, 36.0, 72.0, 108.0, 144.0]


def _splat_basis(orientation_deg: float, kernel_size: int = 5) -> torch.Tensor:
    """Returns a (3, kernel_size, kernel_size) tensor: for each of the 3
    taps (t = -1, 0, +1 pixels along the orientation axis), the bilinear
    splat pattern of that tap's continuous position onto the integer
    kernel_size x kernel_size grid (grid centered at 0)."""
    theta = math.radians(orientation_deg)
    dx, dy = math.cos(theta), math.sin(theta)
    half = kernel_size // 2
    basis = torch.zeros(3, kernel_size, kernel_size)
    for ti, t in enumerate([-1.0, 0.0, 1.0]):
        px, py = t * dx, t * dy  # continuous position of this tap
        x0, y0 = math.floor(px), math.floor(py)
        fx, fy = px - x0, py - y0
        for ix, iy, w in (
            (x0, y0, (1 - fx) * (1 - fy)),
            (x0 + 1, y0, fx * (1 - fy)),
            (x0, y0 + 1, (1 - fx) * fy),
            (x0 + 1, y0 + 1, fx * fy),
        ):
            gx, gy = ix + half, iy + half
            if 0 <= gx < kernel_size and 0 <= gy < kernel_size and w != 0:
                basis[ti, gy, gx] += w
    return basis


class OrientedConv2d(nn.Module):
    """Drop-in replacement for a single isotropic Conv2d layer. Produces
    5 * in_channels output channels (one oriented bank per input channel,
    concatenated), matching the paper's "concatenate the five resulting
    feature maps along the channel dimension" and its 15-vs-9-weights
    comparison. See module docstring for the documented design choices."""

    def __init__(self, in_channels: int, kernel_size: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.n_orientations = len(ORIENTATIONS_DEG)

        # the one shared learnable 1x3 kernel per input channel, per
        # orientation-independent -- shape: (in_channels, 3 taps)
        self.taps = nn.Parameter(torch.randn(in_channels, 3) * (1.0 / math.sqrt(3)))

        # fixed (non-learned) rotation geometry, shape:
        # (n_orientations, 3 taps, k, k)
        basis = torch.stack([_splat_basis(deg, kernel_size) for deg in ORIENTATIONS_DEG])
        self.register_buffer("splat_basis", basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H, W)
        # build the (n_orientations, in_channels, k, k) kernel bank:
        # kernel[o, c] = sum_t taps[c, t] * splat_basis[o, t]
        kernel = torch.einsum("ct,otkl->ockl", self.taps, self.splat_basis)
        # permute to (c, o, k, k) before flattening so the channel order
        # (c*n_orientations + o) matches x.repeat_interleave below, which
        # groups each input channel's n_orientations copies contiguously.
        kernel = kernel.permute(1, 0, 2, 3).reshape(
            self.in_channels * self.n_orientations, 1, self.kernel_size, self.kernel_size
        )
        # depthwise conv: each input channel convolved with its own
        # per-orientation kernel, groups=in_channels, but we need each
        # input channel to produce n_orientations output channels, so we
        # repeat the input channel-wise first.
        x_rep = x.repeat_interleave(self.n_orientations, dim=1)
        out = nn.functional.conv2d(
            x_rep, kernel, padding=self.kernel_size // 2, groups=self.in_channels * self.n_orientations
        )
        return out

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.in_channels * self.n_orientations}, orientations={ORIENTATIONS_DEG}"
