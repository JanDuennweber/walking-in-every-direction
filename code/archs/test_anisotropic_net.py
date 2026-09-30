import torch

from archs.anisotropic_pixel_embedding_net import AnisotropicPixelEmbeddingNet
from archs.isotropic_pixel_embedding_net import MyPixelEmbeddingNet
from losses.discriminative_loss import DiscriminativeLoss

x = torch.randn(2, 1, 20, 40)

iso = MyPixelEmbeddingNet(dim=16)
out_iso = iso(x)
print("isotropic  output shape:", out_iso.shape)
print("isotropic  param count: ", sum(p.numel() for p in iso.parameters()))

aniso = AnisotropicPixelEmbeddingNet(dim=16)
out_aniso = aniso(x)
print("anisotropic output shape:", out_aniso.shape, "(must match isotropic's for a fair ablation)")
print("anisotropic param count: ", sum(p.numel() for p in aniso.parameters()))

assert out_iso.shape == out_aniso.shape, "embedding shapes diverge -- not a fair comparison"

# full backward pass through the real discriminative loss, same as training
target = (torch.rand(2, 3, 20, 40) > 0.9).float()  # 3 fake instances
loss_fn = DiscriminativeLoss(0.5, 2.5, 2)
loss = loss_fn(out_aniso, target)
loss.backward()
print("discriminative loss (anisotropic):", loss.item())
has_grad = all(p.grad is not None and p.grad.abs().sum().item() > 0
               for p in aniso.parameters() if p.requires_grad)
print("all anisotropic params received gradients:", has_grad)
