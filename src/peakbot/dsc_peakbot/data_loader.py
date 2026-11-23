"""DSC data loading — reads Excel files and manages DSC curves."""

import os
import numpy as np
import pandas as pd

from .core import tic, toc, timeit


class DSCCurve:
    """A single DSC curve (temperature vs heat flow)."""

    def __init__(self, filepath=None, name=None):
        self.filepath = filepath
        self.name = name or (os.path.basename(filepath) if filepath else "unknown")
        self.temperature = None
        self.heat_flow = None

    @timeit
    def load_from_excel(self, filepath=None, temp_col=0, hf_col=1,
                        sheet_name=0, skiprows=0):
        """Load DSC data from an Excel file.

        Args:
            filepath: Path to .xlsx/.xls file.
            temp_col: Column index or name for temperature.
            hf_col: Column index or name for heat flow.
            sheet_name: Sheet to read (default first).
            skiprows: Rows to skip at top of sheet.
        """
        if filepath is not None:
            self.filepath = filepath
            self.name = os.path.basename(filepath)

        df = pd.read_excel(self.filepath, sheet_name=sheet_name,
                           skiprows=skiprows)
        cols = df.columns.tolist()

        # Resolve column indices
        if isinstance(temp_col, int):
            tc = cols[temp_col]
        else:
            tc = temp_col
        if isinstance(hf_col, int):
            hc = cols[hf_col]
        else:
            hc = hf_col

        self.temperature = df[tc].values.astype(np.float32)
        self.heat_flow = df[hc].values.astype(np.float32)

        # Remove NaN rows
        mask = ~(np.isnan(self.temperature) | np.isnan(self.heat_flow))
        self.temperature = self.temperature[mask]
        self.heat_flow = self.heat_flow[mask]

        # Ensure temperature is ascending
        if len(self.temperature) > 1 and self.temperature[0] > self.temperature[-1]:
            self.temperature = self.temperature[::-1]
            self.heat_flow = self.heat_flow[::-1]

        return self

    def to_numpy(self):
        """Return (temperature, heat_flow) as numpy arrays."""
        return self.temperature, self.heat_flow

    @property
    def num_points(self):
        return len(self.temperature) if self.temperature is not None else 0


class DSCDataset:
    """Collection of DSC curves for training or prediction."""

    def __init__(self):
        self.curves = []       # list of DSCCurve
        self.references = []   # list of reference annotation dicts

    @timeit
    def load_folder(self, folder_path, temp_col=0, hf_col=1, **kwargs):
        """Load all Excel files from a folder.

        Args:
            folder_path: Path to folder containing .xlsx/.xls files.
            temp_col: Temperature column index or name.
            hf_col: Heat flow column index or name.
            kwargs: Passed to DSCCurve.load_from_excel.
        """
        exts = ('.xlsx', '.xls')
        files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(exts) and not f.startswith('~')
        ])

        for f in files:
            fpath = os.path.join(folder_path, f)
            try:
                curve = DSCCurve(fpath).load_from_excel(
                    temp_col=temp_col, hf_col=hf_col, **kwargs)
                self.curves.append(curve)
            except Exception as e:
                print(f"  | Warning: could not load '{f}': {e}")

        return self

    @timeit
    def load_references(self, ref_file):
        """Load user-annotated reference peaks from a TSV file.

        Expected columns:
            filename    temp_center  temp_start  temp_end  type
            sample1.xlsx  156.2      145.8       168.3     melt_peak
            sample1.xlsx  80.0       60.0        95.0      baseline

        'type' is one of: melt_peak, baseline, artifact
        """
        self.references = []
        with open(ref_file, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split('\t')
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split('\t')
                ref = {
                    'filename':     parts[0],
                    'temp_center':  float(parts[1]),
                    'temp_start':   float(parts[2]),
                    'temp_end':     float(parts[3]),
                    'type':         parts[4].strip(),
                }
                self.references.append(ref)
        return self

    def get_curve_by_name(self, name):
        for c in self.curves:
            if c.name == name:
                return c
        return None
