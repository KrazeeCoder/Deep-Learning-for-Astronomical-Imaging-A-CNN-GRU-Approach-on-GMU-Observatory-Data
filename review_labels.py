#!/usr/bin/env python3
"""
Multi-image FITS label reviewer (Matplotlib GUI).

- Loads an existing labels CSV (schema: file,label,notes,timestamp)
- Shows a grid of FITS images (default: those currently labeled 'bad') from a folder
- Click a tile to select; press 'g'/'b' to relabel, 'n' to edit note, 's' to skip selection
- Pagination with Left/Right arrows (or buttons). Saves changes back to CSV immediately.

Usage:
    python review_labels.py --folder /path/to/folder --csv labels.csv --filter bad --rows 3 --cols 3

Notes:
    - Images are autoscaled with log1p and percentile clipping for visibility
    - No personal paths are embedded; provide your own --folder and --csv paths
"""

import os
import datetime
import argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from astropy.io import fits


# --------------------------
# Display configuration
# --------------------------
IMG_CMAP = 'gray'
LOG_SCALE = True
PERCENTILE_CLIP = (1, 99)
# Speed configuration
SAMPLE_MAX_PIXELS = 200_000  # maximum pixels used to compute percentiles
THUMBNAIL_MAX_DIM = 512      # max(height, width) for cached thumbnails

# Neutral defaults (no personal information)
DEFAULT_FOLDER = '.'
DEFAULT_CSV = 'labels.csv'


def read_fits_image(path: str) -> np.ndarray:
    """Read FITS primary HDU as a 2D float array (first slice if 3D)."""
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        if data is None:
            raise ValueError(f"No image data in primary HDU: {path}")
        if data.ndim == 3:
            data = data[0]
        return np.array(data, dtype=float)


def autoscale_image(img: np.ndarray, pclip: Tuple[float, float] = PERCENTILE_CLIP, logscale: bool = LOG_SCALE) -> Tuple[np.ndarray, float, float]:
    """Return image prepared for display and vmin/vmax used."""
    # Subsample large images for percentile computation to reduce CPU
    if img.size > SAMPLE_MAX_PIXELS:
        step = int(np.ceil(np.sqrt(img.size / SAMPLE_MAX_PIXELS)))
        sample = img[::step, ::step]
    else:
        sample = img
    flat = sample.ravel()
    flat = flat[~np.isnan(flat)]
    if flat.size == 0:
        # Avoid errors; show zeros
        disp = np.zeros_like(img, dtype=float)
        return disp, 0.0, 1.0
    lo, hi = np.percentile(flat, [pclip[0], pclip[1]])
    disp = img.copy()
    if logscale:
        disp = np.log1p(np.clip(disp, a_min=0, a_max=None))
        # Recompute on subsample of the transformed image
        if disp.size > SAMPLE_MAX_PIXELS:
            step2 = int(np.ceil(np.sqrt(disp.size / SAMPLE_MAX_PIXELS)))
            sample2 = disp[::step2, ::step2]
        else:
            sample2 = disp
        flat2 = sample2.ravel()
        flat2 = flat2[~np.isnan(flat2)]
        if flat2.size:
            lo, hi = np.percentile(flat2, [pclip[0], pclip[1]])
        else:
            lo, hi = 0.0, 1.0
    return disp, float(lo), float(hi)


def load_labels(csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Normalize columns
        for col in ['file', 'label', 'notes', 'timestamp']:
            if col not in df.columns:
                df[col] = ''
        # Normalize and coerce types
        # - Ensure filenames are basenames (in case CSV had paths)
        # - Strip whitespace and lowercase labels for consistent matching
        df['file'] = df['file'].astype(str).map(lambda s: os.path.basename(s).strip())
        df['label'] = df['label'].astype(str).str.strip().str.lower()
        df['notes'] = df['notes'].fillna('').astype(str)
        df['timestamp'] = df['timestamp'].astype(str)
        return df
    return pd.DataFrame(columns=['file', 'label', 'notes', 'timestamp'])


def ensure_csv_dir(csv_path: str) -> None:
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def save_labels(df: pd.DataFrame, csv_path: str) -> None:
    ensure_csv_dir(csv_path)
    df.to_csv(csv_path, index=False)


class GridReviewer:
    def __init__(self, folder: str, csv_path: str, rows: int, cols: int, show_filter: str = 'bad', exts: Optional[List[str]] = None):
        self.folder = os.path.abspath(folder)
        self.csv_path = csv_path
        self.rows = rows
        self.cols = cols
        self.page = 0
        self.show_filter = show_filter.lower()  # 'bad', 'good', or 'all'
        self.exts = [e.lower() for e in (exts or ['.fits', '.fit', '.fts'])]

        if not os.path.isdir(self.folder):
            raise FileNotFoundError(self.folder)

        # Thumbnail cache directory inside the folder
        self.thumb_dir = os.path.join(self.folder, '.thumbs')
        try:
            os.makedirs(self.thumb_dir, exist_ok=True)
        except Exception:
            # If creation fails, we will fall back to on-the-fly rendering
            self.thumb_dir = None

        # Load labels and derive the list to show
        self.df = load_labels(self.csv_path)
        self.files_all = sorted([f for f in os.listdir(self.folder) if os.path.splitext(f)[1].lower() in self.exts])
        self.files_by_label = self._build_files_by_label()
        self.display_list = self._compute_display_list()

        # Selection state (row, col) within current page
        self.selected_idx = 0  # linear index within current page tiles

        # Build figure with a grid of axes plus control buttons
        self.fig = plt.figure(figsize=(4 * self.cols, 4 * self.rows + 1.5))
        self.axes = []  # type: List
        grid_h = 0.85
        grid_w = 0.95
        left = 0.03
        bottom = 0.12
        ax_w = grid_w / self.cols
        ax_h = grid_h / self.rows
        for r in range(self.rows):
            row_axes = []
            for c in range(self.cols):
                ax = self.fig.add_axes([left + c * ax_w, bottom + (self.rows - 1 - r) * ax_h, ax_w * 0.96, ax_h * 0.96])
                ax.set_axis_off()
                row_axes.append(ax)
            self.axes.append(row_axes)

        # Control buttons
        self.ax_prev = self.fig.add_axes([0.03, 0.02, 0.10, 0.06])
        self.ax_next = self.fig.add_axes([0.15, 0.02, 0.10, 0.06])
        self.ax_good = self.fig.add_axes([0.40, 0.02, 0.10, 0.06])
        self.ax_bad  = self.fig.add_axes([0.52, 0.02, 0.10, 0.06])
        self.ax_save = self.fig.add_axes([0.75, 0.02, 0.20, 0.06])
        self.btn_prev = Button(self.ax_prev, 'Prev [Left]')
        self.btn_next = Button(self.ax_next, 'Next [Right]')
        self.btn_good = Button(self.ax_good, 'Mark Good [g]')
        self.btn_bad  = Button(self.ax_bad,  'Mark Bad [b]')
        self.btn_save = Button(self.ax_save, 'Save CSV [Ctrl+S]')

        self.btn_prev.on_clicked(lambda e: self._page_delta(-1))
        self.btn_next.on_clicked(lambda e: self._page_delta(1))
        self.btn_good.on_clicked(lambda e: self._relabel_selected('good'))
        self.btn_bad.on_clicked(lambda e: self._relabel_selected('bad'))
        self.btn_save.on_clicked(lambda e: self._save())

        # Connect key and click handlers
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.cid_close = self.fig.canvas.mpl_connect('close_event', self._on_close)

        self._render_page()
        plt.show()

    # --------------------------
    # Data helpers
    # --------------------------
    def _build_files_by_label(self) -> pd.DataFrame:
        """Return a DataFrame indexed by file (exactly those in folder) with current label and notes.
        This anchors labels to files actually present in the selected folder and preserves their order.
        """
        df = self.df.copy()
        # Keep only the latest label per file if duplicates exist
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.sort_values('timestamp').drop_duplicates('file', keep='last')
            mapping = df.set_index('file')[[ 'label', 'notes', 'timestamp' ]]
        else:
            mapping = pd.DataFrame(columns=['label', 'notes', 'timestamp']).set_index(pd.Index([], name='file'))

        # Build an index exactly matching files in the folder (order preserved)
        idx = pd.Index(self.files_all, name='file')
        labels = mapping.reindex(idx)
        # Fill defaults for files with no prior labels
        if 'label' not in labels.columns:
            labels['label'] = ''
        else:
            labels['label'] = labels['label'].fillna('')
        if 'notes' not in labels.columns:
            labels['notes'] = ''
        else:
            labels['notes'] = labels['notes'].fillna('')
        # Leave timestamp as NaT for unlabeled; will be filled on save
        if 'timestamp' not in labels.columns:
            labels['timestamp'] = pd.NaT
        return labels

    def _compute_display_list(self) -> List[str]:
        if self.show_filter == 'all':
            return list(self.files_by_label.index)
        elif self.show_filter == 'good':
            return [f for f, row in self.files_by_label.iterrows() if str(row.get('label', '')) == 'good']
        else:
            return [f for f, row in self.files_by_label.iterrows() if str(row.get('label', '')) == 'bad']

    # --------------------------
    # Rendering and interaction
    # --------------------------
    def _thumb_path(self, fname: str) -> Optional[str]:
        if not self.thumb_dir:
            return None
        # Keep extension to avoid collisions, append .npy
        return os.path.join(self.thumb_dir, f"{fname}.npy")

    def _thumb_is_fresh(self, fname: str) -> bool:
        tpath = self._thumb_path(fname)
        if not tpath or not os.path.exists(tpath):
            return False
        fpath = os.path.join(self.folder, fname)
        try:
            return os.path.getmtime(tpath) >= os.path.getmtime(fpath)
        except Exception:
            return False

    def _make_thumbnail(self, img: np.ndarray) -> np.ndarray:
        disp, lo, hi = autoscale_image(img)
        # Normalize to 0..1 for fast display
        scale = hi - lo if hi > lo else 1.0
        thumb = (disp - lo) / scale
        thumb = np.clip(thumb, 0.0, 1.0)
        # Downsample by integer step to fit within THUMBNAIL_MAX_DIM
        h, w = thumb.shape
        max_dim = max(h, w)
        if max_dim > THUMBNAIL_MAX_DIM:
            step = int(np.ceil(max_dim / THUMBNAIL_MAX_DIM))
            thumb = thumb[::step, ::step]
        return thumb.astype(np.float32, copy=False)

    def _load_or_build_thumb(self, fname: str) -> np.ndarray:
        fpath = os.path.join(self.folder, fname)
        # Try cache
        if self._thumb_is_fresh(fname):
            try:
                tpath = self._thumb_path(fname)
                if tpath:
                    return np.load(tpath)
            except Exception:
                pass
        # Build from FITS
        img = read_fits_image(fpath)
        thumb = self._make_thumbnail(img)
        # Save cache best-effort
        tpath = self._thumb_path(fname)
        if tpath:
            try:
                np.save(tpath, thumb)
            except Exception:
                pass
        return thumb

    def _render_page(self) -> None:
        total = len(self.display_list)
        tiles_per_page = self.rows * self.cols
        start = self.page * tiles_per_page
        end = min(start + tiles_per_page, total)
        files = self.display_list[start:end]
        self.fig.suptitle(f"Reviewing {self.show_filter} files: {start+1}-{end} / {total}  |  Page {self.page+1}/{max(1, (total + tiles_per_page - 1)//tiles_per_page)}")
        # Clear all axes first
        for r in range(self.rows):
            for c in range(self.cols):
                ax = self.axes[r][c]
                ax.cla()
                ax.set_axis_off()

        # Draw tiles
        for idx, fname in enumerate(files):
            r = idx // self.cols
            c = idx % self.cols
            ax = self.axes[r][c]
            fpath = os.path.join(self.folder, fname)
            title = fname
            try:
                thumb = self._load_or_build_thumb(fname)
                ax.imshow(thumb, origin='lower', cmap=IMG_CMAP, vmin=0.0, vmax=1.0)
            except Exception as e:
                # Show a placeholder with error text
                ax.text(0.5, 0.5, f"Error\n{e}", ha='center', va='center', transform=ax.transAxes, color='red')
                title = f"{fname} (error)"
            # Determine label and notes
            row = self.files_by_label.loc[fname]
            label = str(row.get('label', ''))
            notes = str(row.get('notes', ''))
            # Annotate border color by label
            color = 'lime' if label == 'good' else ('red' if label == 'bad' else 'yellow')
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.0)
            # Add title with label
            if notes:
                ax.set_title(f"{title}\n[{label}] — {notes}", fontsize=9)
            else:
                ax.set_title(f"{title}  [{label}]", fontsize=9)
            ax.set_axis_on()  # show border

        # Highlight selected tile
        self._highlight_selection()
        self.fig.canvas.draw_idle()

    def _highlight_selection(self) -> None:
        tiles_per_page = self.rows * self.cols
        sel = self.selected_idx
        for i in range(tiles_per_page):
            r = i // self.cols
            c = i % self.cols
            ax = self.axes[r][c]
            for spine in ax.spines.values():
                lw = 3.0 if i == sel else 2.0
                spine.set_linewidth(lw)

    def _on_click(self, event) -> None:
        if event.inaxes is None:
            return
        # Find which tile was clicked
        for i in range(self.rows * self.cols):
            r = i // self.cols
            c = i % self.cols
            if event.inaxes == self.axes[r][c]:
                self.selected_idx = i
                self._highlight_selection()
                self.fig.canvas.draw_idle()
                return

    def _on_key(self, event) -> None:
        key = (event.key or '').lower()
        if key in ('left',):
            self._page_delta(-1)
        elif key in ('right',):
            self._page_delta(1)
        elif key == 'g':
            self._relabel_selected('good')
        elif key == 'b':
            self._relabel_selected('bad')
        elif key == 'n':
            self._edit_note_selected()
        elif key == 'ctrl+s':
            self._save()

    def _page_delta(self, delta: int) -> None:
        total = len(self.display_list)
        tiles = self.rows * self.cols
        num_pages = max(1, (total + tiles - 1) // tiles)
        self.page = (self.page + delta) % num_pages
        self.selected_idx = 0
        self._render_page()

    def _selected_filename(self) -> Optional[str]:
        total = len(self.display_list)
        tiles = self.rows * self.cols
        start = self.page * tiles
        end = min(start + tiles, total)
        files = self.display_list[start:end]
        if not files:
            return None
        idx = min(max(0, self.selected_idx), len(files) - 1)
        return files[idx]

    def _relabel_selected(self, new_label: str) -> None:
        fname = self._selected_filename()
        if not fname:
            return
        # Update label row
        self.files_by_label.loc[fname, 'label'] = new_label
        self.files_by_label.loc[fname, 'timestamp'] = datetime.datetime.utcnow()
        # Persist to CSV (merge into main df by file)
        self._persist_labels()
        # If filtering by a specific label, recompute list and keep on same page if possible
        self.display_list = self._compute_display_list()
        self.selected_idx = 0
        self._render_page()

    def _edit_note_selected(self) -> None:
        fname = self._selected_filename()
        if not fname:
            return
        # Blocking text input via GUI is tricky in MPL; fall back to console input
        try:
            print(f"Enter note for {fname} (empty to clear): ", end='', flush=True)
            note = input()
        except Exception:
            note = ''
        self.files_by_label.loc[fname, 'notes'] = (note or '')
        self.files_by_label.loc[fname, 'timestamp'] = datetime.datetime.utcnow()
        self._persist_labels()
        self._render_page()

    def _persist_labels(self) -> None:
        # Merge back into self.df keeping latest per file
        updated = self.files_by_label.reset_index()
        # Coerce timestamp to ISO string
        updated['timestamp'] = pd.to_datetime(updated['timestamp'], errors='coerce').fillna(pd.Timestamp.utcnow())
        updated['timestamp'] = updated['timestamp'].dt.tz_localize(None)
        updated['timestamp'] = updated['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')

        # If original df had duplicates, keep most recent
        merged = updated.copy()
        save_labels(merged[['file', 'label', 'notes', 'timestamp']], self.csv_path)

    def _save(self) -> None:
        self._persist_labels()
        print("Saved labels to", self.csv_path)

    def _on_close(self, _event) -> None:
        # Save on close to be safe
        try:
            self._persist_labels()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Review and relabel FITS files in a grid.")
    parser.add_argument('--folder', default=DEFAULT_FOLDER, help=f'Folder containing FITS files (default: {DEFAULT_FOLDER})')
    parser.add_argument('--csv', default=DEFAULT_CSV, help=f'Path to labels CSV to read/write (default: {DEFAULT_CSV})')
    parser.add_argument('--filter', default='bad', choices=['bad', 'good', 'all'], help='Which labeled files to show')
    parser.add_argument('--rows', type=int, default=3, help='Grid rows')
    parser.add_argument('--cols', type=int, default=3, help='Grid cols')
    parser.add_argument('--exts', nargs='*', default=['.fits', '.fit', '.fts'], help='Extensions to include')
    args = parser.parse_args()

    GridReviewer(folder=args.folder, csv_path=args.csv, rows=args.rows, cols=args.cols, show_filter=args.filter, exts=args.exts)


if __name__ == '__main__':
    main()


