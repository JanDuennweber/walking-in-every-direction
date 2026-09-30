import itertools
import torch
from oriented_conv2d import OrientedConv2d, ORIENTATIONS_DEG

layer = OrientedConv2d(in_channels=1)
n_params = sum(p.numel() for p in layer.parameters())
print("learnable params:", n_params, "(expect 3, one 1x3 kernel per input channel)")
print("weights per input channel (5 orientations x 3 taps):", 5 * 3)

x = torch.randn(2, 1, 20, 40, requires_grad=True)
out = layer(x)
print("output shape:", out.shape, "(expect (2, 5, 20, 40))")
loss = out.sum()
loss.backward()
print("grad flows to input:", x.grad is not None and x.grad.abs().sum().item() > 0)
print("grad flows to taps:", layer.taps.grad is not None and layer.taps.grad.abs().sum().item() > 0)

with torch.no_grad():
    k = torch.einsum("ct,otkl->ockl", layer.taps, layer.splat_basis)[:, 0]
    kernels = [k[o].flatten() for o in range(5)]
for a, b in itertools.combinations(range(5), 2):
    diff = (kernels[a] - kernels[b]).abs().sum().item()
    print(f"orientation {ORIENTATIONS_DEG[a]:.0f} vs {ORIENTATIONS_DEG[b]:.0f}: L1 diff = {diff:.4f}")
