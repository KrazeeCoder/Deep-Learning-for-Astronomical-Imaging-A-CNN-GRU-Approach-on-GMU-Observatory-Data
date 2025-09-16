"""
Standalone training script for the joint frame + sequence model (no SageMaker/AWS).

This script defines the exact model architecture used for training and provides
a normal CLI for running on a local machine (CPU or single/multi-GPU).

Usage (examples):
  python train_joint_model.py \
    --mixed-dir "PATH/TO/mixed_frames" \
    --seq-dirs "PATH/TO/toi5443,PATH/TO/toi5595,PATH/TO/toi5944" \
    --label-csv-primary PATH/TO/mixed_labels.csv \
    --extra-label-dirs "PATH/TO/labels5443,PATH/TO/labels5595,PATH/TO/labels5944" \
    --epochs 20 --frame-batch 32 --seq-batch 8 --img-size 128 \
    --model-out joint_frame_seq.pt

Notes:
- Optimizer: Adam with lr=1e-3, weight_decay=1e-4 (same as training runs)
- Loss: BCEWithLogitsLoss with class imbalance handling (pos_weight)
- Data: FITS files. Labels are loaded from CSVs; if a file is missing a label
  we optionally fall back to filename heuristics: files containing "bad" are 1, others 0.
"""

import os
import re
import csv
import glob
import argparse
from collections import OrderedDict
from typing import List, Tuple, Optional, Dict

import numpy as np
from astropy.io import fits

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =============================
# Global configuration
# =============================
img_size: int = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# =============================
# Label map loading (CSV helpers)
# =============================
def _normalize_key(name: str) -> str:
    return os.path.basename(str(name)).strip().lower()


def _normalize_key_stripped(name: str) -> str:
    base = _normalize_key(name)
    return re.sub(r"^[0-9_]+", "", base)


def load_label_map(primary_csv: Optional[str] = None, extra_label_dirs: Optional[List[str]] = None) -> Dict[str, int]:
    """Load labels from CSV files into a dict: {basename_lower: 0/1}.

    The loader is resilient to varying column names, attempting common
    alternatives for the file and label columns.
    """
    label_map: Dict[str, int] = {}

    def _load_csv(path: str) -> None:
        if not path or not os.path.exists(path):
            return
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = [fn.strip().lower() for fn in (reader.fieldnames or [])]

            def find_col(cands: List[str]) -> Optional[str]:
                for cand in cands:
                    if cand in fieldnames:
                        idx = fieldnames.index(cand)
                        return reader.fieldnames[idx]
                return None

            file_col = find_col(["file", "filename", "path", "image", "fname"])
            label_col = find_col(["label", "class", "target", "y"])
            if not file_col or not label_col:
                return

            for row in reader:
                file_val = str(row.get(file_col, "")).strip()
                label_val = str(row.get(label_col, "")).strip().lower()
                if not file_val:
                    continue
                y = 1 if label_val in ("1", "bad", "true", "yes") else 0
                k1 = _normalize_key(file_val)
                k2 = _normalize_key_stripped(file_val)
                label_map[k1] = y
                label_map[k2] = y

    if primary_csv:
        _load_csv(primary_csv)
    if extra_label_dirs:
        for d in extra_label_dirs:
            if not d:
                continue
            for csv_path in glob.glob(os.path.join(d, "**", "*labels*.csv"), recursive=True):
                _load_csv(csv_path)

    return label_map


# =============================
# FITS IO and datasets
# =============================
def read_fits_tensor(path: str) -> torch.Tensor:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float32)
    med = float(np.nanmedian(data))
    data = np.nan_to_num(data, nan=med)
    diff = data - med
    mad = float(np.median(np.abs(diff)))
    scale = 1.4826 * mad + 1e-6
    data = diff / scale
    x = torch.tensor(data).unsqueeze(0).unsqueeze(0)
    x = torch.nn.functional.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return x.squeeze(0)


class FrameDataset(Dataset):
    def __init__(self, root_dirs: List[str], label_map: Optional[Dict[str, int]] = None, augment=None, use_filename_fallback: bool = True):
        self.files: List[str] = []
        self.labels: List[int] = []
        self.label_map = label_map or {}
        self.augment = augment
        self.use_filename_fallback = use_filename_fallback

        for root in root_dirs:
            if not root or not os.path.isdir(root):
                continue
            for dirpath, _dirnames, filenames in os.walk(root):
                for fname in filenames:
                    lower = fname.lower()
                    if not (lower.endswith('.fits') or lower.endswith('.fits.gz')):
                        continue
                    fpath = os.path.join(dirpath, fname)
                    base = os.path.basename(fname)
                    key = os.path.basename(base).strip().lower()
                    key2 = re.sub(r"^[0-9_]+", "", key)
                    label = None
                    if key in self.label_map:
                        label = self.label_map[key]
                    elif key2 in self.label_map:
                        label = self.label_map[key2]
                    elif self.use_filename_fallback:
                        label = 1 if "bad" in key else 0
                    if label is None:
                        continue
                    self.files.append(fpath)
                    self.labels.append(int(label))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = read_fits_tensor(self.files[idx])
        if callable(self.augment):
            x = self.augment(x)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


class SequenceDataset(Dataset):
    def __init__(self, seq_dirs: List[str], window: int = 9, step: int = 1, label_map: Optional[Dict[str, int]] = None, augment=None, use_filename_fallback: bool = True):
        self.samples: List[Tuple[str, List[str], int]] = []
        self.window = window
        self.step = step
        self.label_map = label_map or {}
        self.augment = augment
        self.use_filename_fallback = use_filename_fallback

        for root in seq_dirs:
            if not root or not os.path.isdir(root):
                continue
            all_files: List[str] = []
            for dirpath, _dirnames, filenames in os.walk(root):
                for fname in filenames:
                    lower = fname.lower()
                    if lower.endswith('.fits') or lower.endswith('.fits.gz'):
                        all_files.append(os.path.join(dirpath, fname))
            all_files.sort()
            n = len(all_files)
            if n < window:
                continue
            rel_files = [os.path.relpath(p, root) for p in all_files]
            for s in range(0, max(0, n - window + 1), step):
                self.samples.append((root, rel_files, s))

    def __len__(self) -> int:
        return len(self.samples)

    def _label_for(self, base: str) -> Optional[int]:
        key = os.path.basename(base).strip().lower()
        key2 = re.sub(r"^[0-9_]+", "", key)
        if key in self.label_map:
            return int(self.label_map[key])
        if key2 in self.label_map:
            return int(self.label_map[key2])
        if self.use_filename_fallback:
            return 1 if "bad" in key else 0
        return None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        root, files, s = self.samples[idx]
        chunk = files[s:s + self.window]
        stack = []
        for fname in chunk:
            x = read_fits_tensor(os.path.join(root, fname))
            stack.append(x)
        x = torch.stack(stack, dim=0)
        if callable(self.augment):
            x = self.augment(x)
        center = self.window // 2
        y_val = self._label_for(chunk[center])
        if y_val is None:
            y_val = 0
        y = torch.tensor(y_val, dtype=torch.float32)
        return x, y


# =============================
# Model definitions
# =============================
 


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = torch.mean(x, dim=(2, 3))
        s = self.fc(y).view(b, c, 1, 1)
        return x * s


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg_out, max_out], dim=1)
        m = self.sigmoid(self.conv(a))
        return x * m


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.down = None
        if in_ch != out_ch:
            self.down = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False), nn.BatchNorm2d(out_ch))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv(x)
        if self.down is not None:
            identity = self.down(x)
        out = out + identity
        return self.relu(out)


class SpatialEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block1 = ResidualBlock(32, 64, dilation=1)
        self.se1 = SEBlock(64)
        self.block2 = ResidualBlock(64, 128, dilation=2)
        self.se2 = SEBlock(128)
        self.spatial_attn = SpatialAttention(kernel_size=7)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = self.block1(z)
        z = self.se1(z)
        z = self.block2(z)
        z = self.se2(z)
        z = self.spatial_attn(z)
        z = self.gap(z).flatten(1)
        return z


class JointFrameSeqModel(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.encoder = SpatialEncoder()
        self.frame_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
        self.gru = nn.GRU(input_size=128, hidden_size=hidden, batch_first=True)
        self.seq_head = nn.Linear(hidden, 1)

    def forward_frame(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.frame_head(z).squeeze(-1)

    def forward_seq(self, x: torch.Tensor) -> torch.Tensor:
        bsz, timesteps, channels, height, width = x.shape
        x = x.view(bsz * timesteps, channels, height, width)
        z = self.encoder(x)
        z = z.view(bsz, timesteps, -1)
        out, _ = self.gru(z)
        logits = self.seq_head(out)
        return logits.squeeze(-1)


# =============================
# Training
# =============================
def _compute_pos_weight(pos: int, neg: int) -> float:
    if pos <= 0:
        return 1.0
    return float(max(1, neg)) / float(max(1, pos))


def _build_sampler(files: List[str], labels: List[int]) -> torch.DoubleTensor:
    """Weighted sampling balancing positive/negative and roughly equalizing per-TOI mix.
    TOI detection is heuristic based on folder names containing substrings like 'toi5352'.
    """
    weights: List[float] = []
    toi_freq: Dict[str, int] = {}
    for f in files:
        low = f.lower()
        if 'toi5352' in low:
            key = 'toi5352'
        elif 'toi5907' in low:
            key = 'toi5907'
        elif 'toi6397' in low:
            key = 'toi6397'
        else:
            key = 'other'
        toi_freq[key] = toi_freq.get(key, 0) + 1
    toi_inv = {k: 1.0 / max(1, v) for k, v in toi_freq.items()}

    pos = float(sum(labels))
    neg = float(len(labels) - sum(labels))
    w_pos = 0.5 / max(1.0, pos)
    w_neg = 0.5 / max(1.0, neg)
    for f, y in zip(files, labels):
        low = f.lower()
        if 'toi5352' in low:
            key = 'toi5352'
        elif 'toi5907' in low:
            key = 'toi5907'
        elif 'toi6397' in low:
            key = 'toi6397'
        else:
            key = 'other'
        w_toi = toi_inv.get(key, 1.0)
        w_cls = w_pos if int(y) == 1 else w_neg
        weights.append(w_toi * w_cls)
    return torch.DoubleTensor(weights)


def train_joint(
    mixed_dirs: List[str],
    seq_dirs: List[str],
    label_csv_primary: Optional[str] = None,
    extra_label_dirs: Optional[List[str]] = None,
    window: int = 9,
    epochs: int = 20,
    frame_batch: int = 32,
    seq_batch: int = 8,
    lam_frame: float = 0.8,
    lam_seq: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 0,
    model_out: Optional[str] = None,
    seq_neighbor_span: int = 1,
    seq_neighbor_decay: float = 0.5,
) -> JointFrameSeqModel:
    label_map = load_label_map(label_csv_primary, extra_label_dirs)

    # Augmentation
    def default_augment(x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-1])
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-2])
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            x = torch.rot90(x, k=k, dims=(-2, -1))
        alpha = 0.9 + 0.2 * torch.rand(1).item()
        beta = 0.05 * (torch.rand(1).item() - 0.5)
        x = x * alpha + beta
        if torch.rand(1).item() < 0.5:
            x = x + 0.01 * torch.randn_like(x)
        return x

    frame_ds = FrameDataset(mixed_dirs + seq_dirs, label_map=label_map, augment=default_augment, use_filename_fallback=True)
    seq_ds = SequenceDataset(seq_dirs, window=window, step=1, label_map=label_map, augment=default_augment, use_filename_fallback=True)

    print(f"Device: {device}")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        print(f"CUDA GPUs: {gpu_count} | {', '.join(names)}")
    else:
        print("CUDA not available; running on CPU")
    print(f"Loaded {len(label_map)} labels from CSVs")
    print(f"Frame dataset size: {len(frame_ds)}")
    frame_pos = int(np.sum(frame_ds.labels)) if len(frame_ds) > 0 else 0
    frame_neg = max(0, len(frame_ds) - frame_pos)
    if len(frame_ds) > 0:
        print(f"Frame labels: bad={frame_pos} good={frame_neg}")
    print(f"Sequence windows: {len(seq_ds)} (window={window}, step=1)")

    # Sequence label distribution (center label)
    seq_pos = 0
    if len(seq_ds) > 0:
        for (root, files, s) in seq_ds.samples:
            center = seq_ds.window // 2
            base = files[s + center]
            key = os.path.basename(base).strip().lower()
            key2 = re.sub(r"^[0-9_]+", "", key)
            if key in label_map:
                y = label_map[key]
            elif key2 in label_map:
                y = label_map[key2]
            else:
                y = 1 if "bad" in key else 0
            seq_pos += int(y)
    seq_neg = max(0, len(seq_ds) - seq_pos)
    if len(seq_ds) > 0:
        print(f"Sequence labels (center): bad={seq_pos} good={seq_neg}")

    pin_mem = torch.cuda.is_available()
    common_loader_args = dict(shuffle=True, num_workers=max(0, int(num_workers)), pin_memory=pin_mem, persistent_workers=(num_workers > 0))
    frame_args = dict(common_loader_args)
    seq_args = dict(common_loader_args)
    if num_workers > 0:
        frame_args["prefetch_factor"] = 4
        seq_args["prefetch_factor"] = 2

    # Class/TOI-balanced sampling
    frame_sampler = torch.utils.data.WeightedRandomSampler(_build_sampler(frame_ds.files, frame_ds.labels), num_samples=len(frame_ds), replacement=True)
    frame_args['shuffle'] = False
    frame_loader = DataLoader(frame_ds, batch_size=frame_batch, sampler=frame_sampler, **frame_args)
    seq_loader = DataLoader(seq_ds, batch_size=seq_batch, **seq_args)

    model = JointFrameSeqModel(hidden=64).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
        print(f"DataParallel enabled for encoder across {torch.cuda.device_count()} GPUs")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    pos_weight_frame = _compute_pos_weight(frame_pos, frame_neg)
    pos_weight_seq = _compute_pos_weight(seq_pos, seq_neg)
    print(f"pos_weight (frame)={pos_weight_frame:.4f} | pos_weight (seq)={pos_weight_seq:.4f}")
    criterion_frame = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_frame, dtype=torch.float32, device=device))
    criterion_seq = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_seq, dtype=torch.float32, device=device))

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val: Optional[float] = None
    no_improve = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        frame_iter = iter(frame_loader)
        seq_iter = iter(seq_loader)
        steps = max(len(frame_loader), len(seq_loader))
        total_loss = 0.0
        count = 0
        sum_loss_f, sum_loss_s = 0.0, 0.0
        count_f, count_s = 0, 0

        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            non_block = torch.cuda.is_available()
            loss = None
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                try:
                    xf, yf = next(frame_iter)
                    xf = xf.to(device, non_blocking=non_block)
                    yf = yf.to(device, non_blocking=non_block)
                    logits_f = model.forward_frame(xf)
                    loss_f = criterion_frame(logits_f, yf)
                    loss = (lam_frame * loss_f) if loss is None else (loss + lam_frame * loss_f)
                    sum_loss_f += float(loss_f)
                    count_f += 1
                except StopIteration:
                    pass
                try:
                    xs, ys = next(seq_iter)
                    xs = xs.to(device, non_blocking=non_block)
                    ys = ys.to(device, non_blocking=non_block)
                    logits_s = model.forward_seq(xs)
                    center = xs.shape[1] // 2
                    k = int(max(0, seq_neighbor_span))
                    weights: List[float] = []
                    losses: List[torch.Tensor] = []
                    for off in range(-k, k + 1):
                        t = center + off
                        if t < 0 or t >= logits_s.shape[1]:
                            continue
                        w = 1.0 if off == 0 else float(pow(max(0.0, min(1.0, seq_neighbor_decay)), abs(off)))
                        l = criterion_seq(logits_s[:, t], ys)
                        weights.append(w)
                        losses.append(w * l)
                    if losses:
                        loss_s = sum(losses) / max(1e-8, sum(weights))
                        loss = (lam_seq * loss_s) if loss is None else (loss + lam_seq * loss_s)
                        sum_loss_s += float(loss_s)
                        count_s += 1
                except StopIteration:
                    pass

            if loss is None:
                continue
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach())
            count += 1

        avg_total = total_loss / max(1, count)
        avg_f = (sum_loss_f / max(1, count_f)) if count_f > 0 else 0.0
        avg_s = (sum_loss_s / max(1, count_s)) if count_s > 0 else 0.0
        print(f"Epoch {epoch + 1}/{epochs} | loss={avg_total:.4f} | frame_loss={avg_f:.4f} | seq_loss={avg_s:.4f} | steps={count}")
        scheduler.step(avg_total)

        if best_val is None or avg_total < best_val - 1e-4:
            best_val = avg_total
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 4:
            print(f"Early stopping: no improvement for {no_improve} epochs")
            break

    # Save model (strip DataParallel prefixes if present)
    state = model.state_dict()
    cleaned = OrderedDict()
    for k, v in state.items():
        cleaned[k.replace('encoder.module', 'encoder')] = v
    out_path = model_out or os.path.join(os.getcwd(), "joint_frame_seq.pt")
    torch.save(cleaned, out_path)
    print(f"Saved model to {out_path}")
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the joint frame + sequence model on local machine.")
    p.add_argument('--mixed-dir', type=str, default=None, help='Path to mixed frames directory')
    p.add_argument('--seq-dirs', type=str, default=None, help='Comma-separated sequence directories (defaults to --mixed-dir if omitted)')
    p.add_argument('--label-csv-primary', type=str, default=None, help='Path to primary labels CSV (e.g., mixed_labels.csv)')
    p.add_argument('--extra-label-dirs', type=str, default=None, help='Comma-separated directories containing *labels*.csv')
    p.add_argument('--window', type=int, default=9)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--frame-batch', type=int, default=32)
    p.add_argument('--seq-batch', type=int, default=8)
    p.add_argument('--lam-frame', type=float, default=0.8)
    p.add_argument('--lam-seq', type=float, default=0.2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--img-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=0, help='DataLoader workers (default 0 for portability)')
    p.add_argument('--model-out', type=str, default='joint_frame_seq.pt', help='Path to save model weights (.pt)')
    return p.parse_args()


def _split_csv(arg: Optional[str]) -> List[str]:
    if not arg:
        return []
    return [p.strip() for p in str(arg).split(',') if p.strip()]


if __name__ == "__main__":
    args = parse_args()
    img_size = int(args.img_size)

    mixed_dirs: List[str] = []
    seq_dirs: List[str] = []
    if args.mixed_dir:
        mixed_dirs.append(args.mixed_dir)
    seq_dirs.extend(_split_csv(args.seq_dirs))

    extra_label_dirs = _split_csv(args.extra_label_dirs)

    print("Starting local training...")
    print(f"Mixed dirs: {', '.join(mixed_dirs) if mixed_dirs else '<none>'}")
    print(f"Sequence dirs: {', '.join(seq_dirs) if seq_dirs else '<none>'}")
    print(f"Primary label CSV: {args.label_csv_primary or '<none>'}")
    print(f"Extra label dirs: {', '.join(extra_label_dirs) if extra_label_dirs else '<none>'}")

    train_joint(
        mixed_dirs=mixed_dirs,
        seq_dirs=seq_dirs,
        label_csv_primary=args.label_csv_primary,
        extra_label_dirs=extra_label_dirs,
        window=args.window,
        epochs=args.epochs,
        frame_batch=args.frame_batch,
        seq_batch=args.seq_batch,
        lam_frame=args.lam_frame,
        lam_seq=args.lam_seq,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        model_out=args.model_out,
    )


