"""
Direction-sensitive variant of MyPixelEmbeddingNet (isotropic_pixel_embedding_net.py),
implementing Subsection 3.5 of the paper: the first convolutional layer
(operating on the full-resolution, un-pooled 20x40 input) and the final
layer of the mirrored upsampling branch are replaced with the oriented
kernel bank (OrientedConv2d, 5 kernels at {0,36,72,108,144} degrees);
every other layer is unchanged from the isotropic baseline.

CHANNEL-COUNT CONSEQUENCE (a real design decision, not specified verbatim
by the paper text -- documented here rather than silently assumed): each
OrientedConv2d multiplies its input channel count by 5 (one bank per input
channel, concatenated). Applied to d1 (in_channels=1), this gives 5 output
channels instead of the isotropic version's 4 -- that ripples through
every downstream layer's channel count (see the diagram below). Applied to
the final layer (in_channels=8, matching the isotropic version's
penultimate layer), it would give 40 channels, not the paper's stated
16-dimensional embedding space -- so a final 1x1 projection
(`proj_out`) reduces 40 -> 16 to keep the embedding dimension identical to
the isotropic baseline, which is necessary for a fair, comparable ablation
(Table 3, Subsection 4.3) and for compatibility with the unmodified
discriminative loss.

Channel flow (vs. isotropic_pixel_embedding_net.py in parentheses):
  input (1) -> d1 [OrientedConv2d] -> 5 (was 4)
  pool1 -> d2_conv1 (5->8, was 4->8) -> 8
  pool2 -> d3_conv1 (8->16, unchanged) -> 16
  u1_tconv (16->8, unchanged) -> u1_conv1 (cat 8+8=16 -> 8, unchanged)
  u2_tconv (8->5, was 8->4, changed to match d1's new width)
  u2_conv1 (cat 5+5=10 -> 8, in_channels changed from 8->10)
  conv_out [OrientedConv2d] (8 -> 40) -> proj_out [1x1 conv] (40 -> 16)
"""

import torch
import torch.nn as nn

from .oriented_conv2d import OrientedConv2d


class AnisotropicPixelEmbeddingNet(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()

        # down
        self.d1_conv1 = OrientedConv2d(in_channels=1)  # -> 5 channels
        self.d1_reluconv1 = nn.ReLU()

        self.pool1 = nn.AvgPool2d(2)
        self.d2_conv1 = nn.Conv2d(5, 8, kernel_size=3, padding=1)
        self.d2_reluconv1 = nn.ReLU()

        self.pool2 = nn.AvgPool2d(2)
        self.d3_conv1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.d3_reluconv1 = nn.ReLU()

        # up
        self.u1_tconv = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.u1_conv1 = nn.Conv2d(16, 8, kernel_size=3, padding=1)

        self.u2_tconv = nn.ConvTranspose2d(8, 5, kernel_size=2, stride=2)
        self.u2_conv1 = nn.Conv2d(10, 8, kernel_size=3, padding=1)

        # out: oriented bank (8 -> 40), then project down to the target
        # embedding dimension so it matches the isotropic baseline exactly
        self.conv_out = OrientedConv2d(in_channels=8)  # -> 40 channels
        self.proj_out = nn.Conv2d(40, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: bs x 1 x height x width
        d1 = self.d1_reluconv1(self.d1_conv1(x))
        d2 = self.d2_reluconv1(self.d2_conv1(self.pool1(d1)))
        d3 = self.d3_reluconv1(self.d3_conv1(self.pool2(d2)))

        upconv1 = self.u1_tconv(d3)
        u1 = self.u1_conv1(torch.cat((upconv1, d2), dim=1))
        u2 = self.u2_conv1(torch.cat((self.u2_tconv(u1), d1), dim=1))

        out = self.proj_out(self.conv_out(u2))
        return out
