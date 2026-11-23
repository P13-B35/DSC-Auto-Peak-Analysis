"""Visualization for DSC peak detection — academic publication quality."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots  # noqa — registers style sheets

# ---------------------------------------------------------------------------
# Global style — applied once at import time
# ---------------------------------------------------------------------------
try:
    plt.style.use(['science', 'no-latex'])
except Exception:
    pass  # fall back to matplotlib default if SciencePlots unavailable

# ---------------------------------------------------------------------------
# Colour palette (colour-blind friendly, publication-safe)
# ---------------------------------------------------------------------------
C_MELT     = '#d62728'   # red — melt peak
C_BASELINE = '#1f77b4'   # blue — baseline region
C_ARTIFACT = '#7f7f7f'   # grey — artifact / noise
C_CURVE    = '#262626'   # near-black for the DSC trace
C_SHADE    = '#d62728'   # melt-peak shading (same as melt)
C_BOUNDARY = '#2c3e50'   # dark slate for boundary markers

TYPE_COLORS = {
    'melt_peak': C_MELT,
    'baseline':  C_BASELINE,
    'artifact':  C_ARTIFACT,
}

TYPE_LABELS = {
    'melt_peak': 'Melt peak',
    'baseline':  'Baseline',
    'artifact':  'Artifact',
}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _exothermic_arrow(ax, x, y, label, offset=0.5):
    """Place a small downward arrow + label indicating exothermic direction."""
    y0, y1 = ax.get_ylim()
    dy = (y1 - y0) * 0.03
    ax.annotate(
        label,
        xy=(x, y), xytext=(x, y + offset * dy),
        fontsize=8, ha='center', va='bottom',
        color='#555555',
        arrowprops=dict(arrowstyle='->', color='#555555', lw=0.8),
    )


# ---------------------------------------------------------------------------
# Main plotting
# ---------------------------------------------------------------------------
def plot_detection(curve_path, peaks, output_path, title=None,
                   temp_col='temperature', hf_col='heatflow',
                   figsize=(7.0, 4.0), dpi=300,
                   show_all_peaks=True):
    """Plot one DSC curve with detected peaks overlaid.

    Args:
        curve_path: Path to Excel file.
        peaks: List of peak dicts from predict_on_curve.
        output_path: Where to save the figure (PNG recommended).
        title: Optional title string.
        temp_col: Column name for temperature.
        hf_col: Column name for heat flow.
        figsize: Figure size in inches (default 7x4 = double-column width).
        dpi: Output resolution.
        show_all_peaks: If False, only show melt_peak detections.
    """
    # -- load data ----------------------------------------------------------
    df = pd.read_excel(curve_path)
    temp = df[temp_col].values.astype(np.float64)
    hf   = df[hf_col].values.astype(np.float64)

    # -- set up figure ------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # DSC trace
    ax.plot(temp, hf, color=C_CURVE, linewidth=0.7, label='DSC signal', zorder=2)

    # -- group peaks by type ------------------------------------------------
    melt_peaks  = [p for p in peaks if p['type_name'] == 'melt_peak']
    base_peaks  = [p for p in peaks if p['type_name'] == 'baseline']
    art_peaks   = [p for p in peaks if p['type_name'] == 'artifact']

    # ---- shade melt-peak regions ------------------------------------------
    for p in melt_peaks:
        ts, te = p['temp_start'], p['temp_end']
        if ts > te:
            ts, te = te, ts   # fix inverted boundaries
        ax.axvspan(ts, te, alpha=0.12, color=C_SHADE, linewidth=0, zorder=1)

    # ---- mark boundaries for melt peaks -----------------------------------
    for p in melt_peaks:
        ts, te = p['temp_start'], p['temp_end']
        if ts > te:
            ts, te = te, ts
        ct, ch = p['center_temp'], p['center_hf']

        # vertical dashed lines at boundaries
        ax.axvline(ts, color=C_BOUNDARY, linewidth=0.6, linestyle=(0, (5, 3)),
                   alpha=0.6, zorder=3)
        ax.axvline(te, color=C_BOUNDARY, linewidth=0.6, linestyle=(0, (5, 3)),
                   alpha=0.6, zorder=3)

        # centre marker
        ax.plot(ct, ch, 'o', color=C_MELT, markersize=5,
                markeredgecolor='white', markeredgewidth=0.5, zorder=5)

    # ---- mark other peak types (lighter) -----------------------------------
    if show_all_peaks:
        for p in base_peaks + art_peaks:
            ct, ch = p['center_temp'], p['center_hf']
            c = TYPE_COLORS.get(p['type_name'], C_ARTIFACT)
            ax.plot(ct, ch, 'x', color=c, markersize=4, alpha=0.45,
                    markeredgewidth=0.6, zorder=4)

    # -- legend entries -----------------------------------------------------
    if melt_peaks:
        ax.plot([], [], 'o', color=C_MELT, markersize=5,
                markeredgecolor='white', markeredgewidth=0.5,
                label=f'Melt peak ({len(melt_peaks)})')
    if base_peaks and show_all_peaks:
        ax.plot([], [], 'x', color=C_BASELINE, markersize=4, alpha=0.7,
                label=f'Baseline ({len(base_peaks)})')
    if art_peaks and show_all_peaks:
        ax.plot([], [], 'x', color=C_ARTIFACT, markersize=4, alpha=0.7,
                label=f'Artifact ({len(art_peaks)})')

    # -- labels & formatting ------------------------------------------------
    ax.set_xlabel('Temperature ($^\circ$C)')
    ax.set_ylabel('Heat flow (mW)')

    if title is None:
        title = os.path.basename(curve_path)
    ax.set_title(title, fontsize=11, pad=6)

    ax.legend(loc='upper right', frameon=True, fontsize=7,
              framealpha=0.9, edgecolor='#cccccc')

    # -- zoom hint if melt peaks present ------------------------------------
    if melt_peaks:
        # slightly tighten x-range to show melt region with context
        all_melt_temps = []
        for p in melt_peaks:
            ts, te = p['temp_start'], p['temp_end']
            if ts > te:
                ts, te = te, ts
            all_melt_temps.extend([ts, te])
        if all_melt_temps:
            margin = (max(all_melt_temps) - min(all_melt_temps)) * 0.4
            ax.set_xlim(min(all_melt_temps) - margin,
                        max(all_melt_temps) + margin)

    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Batch plotting
# ---------------------------------------------------------------------------
def plot_all_detections(curve_folder, peaks_by_file, output_dir,
                        temp_col='temperature', hf_col='heatflow',
                        figsize=(7.0, 4.0), dpi=300,
                        show_all_peaks=True, verbose=True):
    """Plot all curves with their detected peaks.

    Args:
        curve_folder: Folder containing Excel files.
        peaks_by_file: Dict filename -> list of peak dicts.
        output_dir: Where to save PNG files.
        temp_col: Temperature column name.
        hf_col: Heat flow column name.
        figsize: Figure size in inches.
        dpi: Output resolution.
        show_all_peaks: If False, only melt peaks are shown.
        verbose: Print progress.
    """
    os.makedirs(output_dir, exist_ok=True)

    for fname, peaks in peaks_by_file.items():
        fpath = os.path.join(curve_folder, fname)
        if not os.path.exists(fpath):
            if verbose:
                print(f"  [WARN] '{fname}' not found, skipping")
            continue

        out_name = os.path.splitext(fname)[0] + '.png'
        out_path = os.path.join(output_dir, out_name)

        plot_detection(fpath, peaks, out_path, title=fname,
                       temp_col=temp_col, hf_col=hf_col,
                       figsize=figsize, dpi=dpi,
                       show_all_peaks=show_all_peaks)

        n_melt = sum(1 for p in peaks if p['type_name'] == 'melt_peak')
        if verbose:
            print(f"  {fname} -> {out_name}  ({n_melt} melt peaks)")


# ---------------------------------------------------------------------------
# Summary figure — all curves in a grid
# ---------------------------------------------------------------------------
def plot_summary_grid(curve_folder, peaks_by_file, output_path,
                      temp_col='temperature', hf_col='heatflow',
                      dpi=300, verbose=True):
    """Create a multi-panel figure with all curves side by side.

    Args:
        curve_folder: Folder containing Excel files.
        peaks_by_file: Dict filename -> list of peak dicts.
        output_path: Where to save the summary figure.
        temp_col: Temperature column name.
        hf_col: Heat flow column name.
        dpi: Output resolution.
        verbose: Print progress.
    """
    n = len(peaks_by_file)
    if n == 0:
        print('[WARN] No curves to plot')
        return

    # layout: ~3 columns
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.5 * nrows),
                             dpi=dpi, squeeze=False)

    for idx, (fname, peaks) in enumerate(peaks_by_file.items()):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        fpath = os.path.join(curve_folder, fname)
        if not os.path.exists(fpath):
            ax.text(0.5, 0.5, f'{fname}\n(file not found)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=8, color='grey')
            continue

        df = pd.read_excel(fpath)
        temp = df[temp_col].values.astype(np.float64)
        hf = df[hf_col].values.astype(np.float64)

        ax.plot(temp, hf, color=C_CURVE, linewidth=0.5)

        # melt peaks only for grid
        for p in peaks:
            if p['type_name'] != 'melt_peak':
                continue
            ts, te = p['temp_start'], p['temp_end']
            if ts > te:
                ts, te = te, ts
            ax.axvspan(ts, te, alpha=0.15, color=C_MELT, linewidth=0)
            ax.plot(p['center_temp'], p['center_hf'], 'o',
                    color=C_MELT, markersize=3,
                    markeredgecolor='white', markeredgewidth=0.4)

        n_melt = sum(1 for p in peaks if p['type_name'] == 'melt_peak')
        ax.set_title(fname, fontsize=8, pad=2)
        ax.set_xlabel('Temp. ($^\circ$C)', fontsize=7)
        ax.set_ylabel('HF (mW)', fontsize=7)
        ax.tick_params(labelsize=6)
        if n_melt:
            ax.text(0.98, 0.95, f'{n_melt} melt',
                    transform=ax.transAxes, fontsize=7,
                    ha='right', va='top', color=C_MELT,
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.8, edgecolor='none'))

    # hide unused subplots
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.tight_layout(pad=0.8)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f'Summary grid saved to {output_path}')
