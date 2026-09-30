#!/usr/bin/env python3
"""
Training entry point for the Table 3 conditions (Subsection 4.3 of the paper).

Reuses Geyer's original training procedure (training_loop.py,
DiscriminativeLoss, MyInstanceSegDataset) unchanged in spirit -- same
optimizer (Adam, default hyperparameters), same loss (DiscriminativeLoss(
0.5, 2.5, 2)), same 40x20 grid convention -- only made configurable
(architecture, data file, output directory) instead of hardcoded.

Usage:
    python3 train.py --train-data data/train_synthetic_old_range.txt \\
        --val-data data/val_synthetic.txt --arch isotropic \\
        --epochs 200 --model-name row2_narrow_isotropic --out-dir results/row2

Table 3 condition -> arch / data mapping (run_table3_final.sh runs all four):
    row 1 (scarce real only)      : --train-data data/real_scarce_train.txt       --arch isotropic
    row 2 (narrow range)          : --train-data data/train_synthetic_old_range.txt --arch isotropic
    row 3 (naive uniform 360)     : --train-data data/train_naive_360.txt         --arch isotropic
    row 4 (Robinson, full method) : --train-data data/train_robinson.txt          --arch anisotropic
"""

import argparse
import os
import random

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from archs.isotropic_pixel_embedding_net import MyPixelEmbeddingNet
from archs.anisotropic_pixel_embedding_net import AnisotropicPixelEmbeddingNet
from losses.discriminative_loss import DiscriminativeLoss
from dataset import MyInstanceSegDataset, collate_fn
from training_loop import training_loop

ARCHS = {
    "isotropic": MyPixelEmbeddingNet,
    "anisotropic": AnisotropicPixelEmbeddingNet,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--arch", choices=list(ARCHS), required=True)
    ap.add_argument("--epochs", type=int, default=200)  # 200 matches the paper's reported training run
    ap.add_argument("--batch-size", type=int, default=16)  # matches Geyer's original InstanceSegmentation.py
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--width", type=int, default=40)
    ap.add_argument("--height", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=None,
                    help="seeds weight init and batch shuffling; default: unseeded, as before")
    args = ap.parse_args()

    loader_generator = None
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)  # also seeds CUDA
        loader_generator = torch.Generator().manual_seed(args.seed)
        print("Seed:", args.seed)

    device = torch.device(args.device)
    print("Device:", device)

    print("Loading training data:", args.train_data)
    train_data = MyInstanceSegDataset(args.train_data, args.width, args.height)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                               drop_last=True, collate_fn=collate_fn, generator=loader_generator)

    print("Loading validation data:", args.val_data)
    val_data = MyInstanceSegDataset(args.val_data, args.width, args.height)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True,
                             drop_last=True, collate_fn=collate_fn)

    model = ARCHS[args.arch]().to(device=device)
    optimizer = optim.Adam(model.parameters())
    loss_fn = DiscriminativeLoss(0.5, 2.5, 2)

    os.makedirs(args.out_dir, exist_ok=True)
    training_loop(
        n_epochs=args.epochs, optimizer=optimizer, model=model, loss_fn=loss_fn,
        train_loader=train_loader, validation_loader=val_loader, device=device,
        model_name=args.model_name, out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
