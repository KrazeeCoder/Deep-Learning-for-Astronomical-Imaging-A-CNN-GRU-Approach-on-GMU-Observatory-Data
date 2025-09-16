"""
Local evaluation for the joint frame + sequence model (no SageMaker, no cloud).

Defaults:
- Uses both heads: frame + sequence(center) with weighted fusion (frame_weight=0.8)
- Threshold=0.5 (can calibrate to maximize a metric on your eval set)

Example:
  python evaluate_local.py \
    --data-dirs PATH/TO/mixed_frames \
    --seq-dirs PATH/TO/toi5443,PATH/TO/toi5595,PATH/TO/toi5944 \
    --model-path joint_frame_seq.pt \
    --combine-policy weighted --frame-weight 0.8 --calibrate-threshold
"""

import os
import argparse
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_joint_model import (
    FrameDataset,
    SequenceDataset,
    JointFrameSeqModel,
    img_size as GLOBAL_IMG_SIZE,
)


def split_csv(arg: Optional[str]) -> List[str]:
    if not arg:
        return []
    return [p.strip() for p in str(arg).split(',') if p.strip()]


def safe_div(n: float, d: float) -> float:
    return (n / d) if d != 0 else 0.0


def compute_confusion(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return tp, tn, fp, fn


def compute_metrics(y_true: List[int], y_pred: List[int]) -> dict:
    tp, tn, fp, fn = compute_confusion(y_true, y_pred)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) != 0 else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local evaluation using both frame and sequence (center) heads.")
    p.add_argument('--model-path', type=str, default='joint_frame_seq.pt', help='Path to model state_dict (.pt/.pth)')
    p.add_argument('--data-dirs', type=str, required=True, help='Comma-separated frame directories to evaluate')
    p.add_argument('--seq-dirs', type=str, default=None, help='Comma-separated sequence directories (defaults to data-dirs)')
    p.add_argument('--img-size', type=int, default=128, help='Image size (must match training)')
    p.add_argument('--batch-size', type=int, default=64, help='Batch size for frame eval')
    p.add_argument('--seq-batch-size', type=int, default=8, help='Batch size for sequence eval')
    p.add_argument('--threshold', type=float, default=0.5, help='Classification threshold on probability')
    p.add_argument('--calibrate-threshold', action='store_true', help='Calibrate threshold on this eval set (maximize F1)')
    p.add_argument('--combine-policy', type=str, default='weighted', choices=['frame_only','seq_center_only','max_prob','avg_prob','weighted'], help='Fusion rule for frame vs seq(center)')
    p.add_argument('--frame-weight', type=float, default=0.8, help='Weight for frame probability in weighted fusion (seq weight = 1 - frame_weight)')
    p.add_argument('--out-csv', type=str, default=None, help='Optional CSV path to write per-file predictions')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure eval resize matches training
    from train_joint_model import img_size as TRAIN_IMG_SIZE
    global GLOBAL_IMG_SIZE
    GLOBAL_IMG_SIZE = int(args.img_size or TRAIN_IMG_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sigmoid = nn.Sigmoid()

    data_dirs = split_csv(args.data_dirs)
    seq_dirs = split_csv(args.seq_dirs) or data_dirs

    # Datasets/loaders
    eval_ds = FrameDataset(data_dirs, label_map=None, augment=None, use_filename_fallback=True)
    if len(eval_ds) == 0:
        raise RuntimeError('No FITS files found in data-dirs')
    seq_ds = SequenceDataset(seq_dirs, window=9, step=1, label_map=None, augment=None, use_filename_fallback=True)

    frame_loader = DataLoader(
        eval_ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, (os.cpu_count() or 0) // 2),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    seq_loader = None
    if len(seq_ds) > 0:
        seq_loader = DataLoader(
            seq_ds,
            batch_size=max(1, int(args.seq_batch_size)),
            shuffle=False,
            num_workers=max(0, (os.cpu_count() or 0) // 2),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
        )

    # Model
    model = JointFrameSeqModel(hidden=64).to(device)
    state = torch.load(args.model_path, map_location=device)
    load_result = model.load_state_dict(state, strict=False)
    if load_result.missing_keys:
        print(f"[warn] Missing keys: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"[warn] Unexpected keys: {load_result.unexpected_keys}")
    model.eval()

    y_prob: List[float] = []
    y_pred: List[int] = []
    # If labels exist in filenames (fallback), we can compute rough metrics; otherwise, only probs are meaningful
    y_true: List[int] = []
    file_paths: List[str] = []

    frame_probs_all: List[float] = []

    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(frame_loader):
            xb = xb.to(device, non_blocking=True)
            logits_f = model.forward_frame(xb)
            probs_f = sigmoid(logits_f).detach().float().cpu().tolist()
            frame_probs_all.extend(probs_f)
            y_true.extend(yb.detach().float().cpu().tolist())
            start = batch_idx * len(probs_f)
            end = min(start + len(probs_f), len(eval_ds.files))
            file_paths.extend(eval_ds.files[start:end])

        seq_center_probs_all: Optional[List[float]] = None
        if seq_loader is not None and len(seq_ds) > 0:
            seq_center_probs_all = []
            for xs, ys in seq_loader:
                xs = xs.to(device, non_blocking=True)
                logits_s = model.forward_seq(xs)
                center = xs.shape[1] // 2
                probs_s = sigmoid(logits_s[:, center]).detach().float().cpu().tolist()
                seq_center_probs_all.extend(probs_s)

    # Combine
    if seq_center_probs_all is not None and len(seq_center_probs_all) == len(frame_probs_all):
        if args.combine_policy == 'frame_only':
            y_prob = frame_probs_all
        elif args.combine_policy == 'seq_center_only':
            y_prob = seq_center_probs_all
        elif args.combine_policy == 'max_prob':
            y_prob = [max(a, b) for a, b in zip(frame_probs_all, seq_center_probs_all)]
        elif args.combine_policy == 'avg_prob':
            y_prob = [0.5 * (a + b) for a, b in zip(frame_probs_all, seq_center_probs_all)]
        else:
            fw = float(max(0.0, min(1.0, args.frame_weight)))
            sw = 1.0 - fw
            y_prob = [fw * a + sw * b for a, b in zip(frame_probs_all, seq_center_probs_all)]
    else:
        y_prob = frame_probs_all

    # Optional threshold calibration
    best_threshold = float(args.threshold)
    if args.calibrate_threshold and len(y_true) == len(y_prob) and len(y_true) > 0:
        def thresholds():
            t = 0.05
            while t <= 0.95 + 1e-8:
                yield round(t, 3)
                t += 0.01
        best_score = -1.0
        for t in thresholds():
            preds = [1 if p >= t else 0 for p in y_prob]
            m = compute_metrics([int(v) for v in y_true], preds)
            if m['f1'] > best_score:
                best_score = m['f1']
                best_threshold = t

    y_pred = [1 if p >= best_threshold else 0 for p in y_prob]

    print(f"Samples: {len(y_prob)}")
    if len(y_true) == len(y_prob) and len(y_true) > 0:
        m = compute_metrics([int(v) for v in y_true], y_pred)
        print("Overall:")
        print(f"  accuracy:  {m['accuracy']:.4f}")
        print(f"  precision: {m['precision']:.4f}")
        print(f"  recall:    {m['recall']:.4f}")
        print(f"  f1:        {m['f1']:.4f}")
    print(f"Threshold used: {best_threshold:.3f}{' (calibrated)' if args.calibrate_threshold else ''}")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True) if os.path.dirname(args.out_csv) else None
        import csv
        with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['file','prob_bad','pred'])
            for fp, pr, yp in zip(file_paths, y_prob, y_pred):
                w.writerow([fp, float(pr), int(yp)])
        print(f"Wrote predictions to {args.out_csv}")


if __name__ == '__main__':
    main()


