#!/usr/bin/env python3
"""
Train an RF-DETR Medium instance-segmentation model for the Frosch bottle dataset.

IMPORTANT:
- This intentionally does NOT load the existing detection checkpoint
  (checkpoint_best_regular.pth).
- RF-DETR detection and segmentation use different architectures.
- The segmentation model starts from its own COCO-pretrained segmentation
  weights and fine-tunes on the existing COCO-segmentation dataset.
"""

from pathlib import Path
import argparse

from rfdetr import RFDETRSegMedium


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=".",
        help="COCO dataset root containing train/valid/test splits.",
    )
    parser.add_argument(
        "--output",
        default="runs/frosch_seg_medium",
        help="Directory for segmentation checkpoints.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=432)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Frosch RF-DETR Segmentation Training")
    print("=" * 60)
    print(f"Dataset       : {dataset}")
    print(f"Output        : {output}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Grad accum    : {args.grad_accum_steps}")
    print(f"Resolution    : {args.resolution}")
    print(f"Learning rate : {args.lr}")
    print()
    print("Using RFDETRSegMedium COCO-pretrained segmentation weights.")
    print("NOT loading runs/frosch_medium/checkpoint_best_regular.pth.")
    print("=" * 60)

    model = RFDETRSegMedium()

    model.train(
        dataset_dir=str(dataset),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        resolution=args.resolution,
        output_dir=str(output),
    )


if __name__ == "__main__":
    main()
