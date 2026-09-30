#!/usr/bin/env python3
"""
Evaluation entry point -- fills in the Precision/Recall columns of
Table 3 (Subsection 4.3) for one trained model.

Adapted from Geyer's original precision_inst_seg.py (which had the same
data/model choices hardcoded, and a genuine syntax error on its first
line -- `import torch from MyPixelEmbeddingNet import ...`, two
statements concatenated onto one line -- fixed here). Same postprocessing
pipeline: mean-shift clustering (bandwidth=2.5, matching the delta_var=0.5
the model was trained with) then IoU-thresholded precision/recall,
computed at the same three thresholds the paper's Table 1/Table 2 use
(0.50, 0.75, 0.95) for direct comparability.

Usage:
    python3 evaluate.py --data data/real_scarce.txt --arch isotropic \\
        --model results/row1/models/row1_real_only.pt --out results/row1/eval.csv
"""

import argparse
import csv

import torch
from torch.utils.data import DataLoader

from archs.isotropic_pixel_embedding_net import MyPixelEmbeddingNet
from archs.anisotropic_pixel_embedding_net import AnisotropicPixelEmbeddingNet
from dataset import MyInstanceSegDataset, collate_fn
from postprocess.postprocess_util import calculate_metrics_for_batch, post_process_model_output

ARCHS = {
    "isotropic": MyPixelEmbeddingNet,
    "anisotropic": AnisotropicPixelEmbeddingNet,
}

IOU_THRESHOLDS = [0.50, 0.75, 0.95]  # matches Table 1 / Table 2 in the paper


def evaluate(model, data_loader, iou_threshold):
    model.eval()
    total_prec, total_recall, n = 0.0, 0.0, 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            model_output = model(inputs)
            processed = post_process_model_output(model_output, inputs)
            precision, recall, _ = calculate_metrics_for_batch(processed, labels, iou_threshold=iou_threshold)
            total_prec += precision
            total_recall += recall
            n += 1
    return total_prec / n, total_recall / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arch", choices=list(ARCHS), required=True)
    ap.add_argument("--model", required=True, help="path to a .pt state_dict")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--width", type=int, default=40)
    ap.add_argument("--height", type=int, default=20)
    ap.add_argument("--condition", default="", help="free-text label for the CSV, e.g. 'row1'")
    args = ap.parse_args()

    print("Loading data:", args.data)
    data = MyInstanceSegDataset(args.data, args.width, args.height)
    # batch_size=1: matches Geyer's original evaluation script; the IoU/
    # clustering postprocessing is per-sample regardless, so this does
    # not change what is measured, only how it is batched.
    loader = DataLoader(data, batch_size=1, shuffle=False, drop_last=True, collate_fn=collate_fn)

    model = ARCHS[args.arch]()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))

    rows = []
    for thr in IOU_THRESHOLDS:
        precision, recall = evaluate(model, loader, thr)
        print(f"IoU={thr}: precision={precision:.4f} recall={recall:.4f}")
        rows.append({"condition": args.condition, "arch": args.arch, "data": args.data,
                     "iou_threshold": thr, "precision": precision, "recall": recall})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "arch", "data", "iou_threshold", "precision", "recall"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
