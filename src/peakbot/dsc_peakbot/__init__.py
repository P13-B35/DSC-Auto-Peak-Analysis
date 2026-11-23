"""DSC PeakBot — Machine-learning based DSC melt-peak detection."""

from .core import tic, toc, timeit, TabLog, readTSVFile, writeTSVFile
from .config import Config
from .data_loader import DSCCurve, DSCDataset
from .preprocessing import (
    find_local_maxima,
    extract_window,
    normalize_window,
    extract_scalar_features,
    find_boundaries_full_curve,
    gradient_descend_boundaries,
    prepare_instances_from_references,
    prepare_instances_for_prediction,
    save_instances,
    construct_baseline_iterative_poly,
    calculate_enthalpy,
    calculate_onset_endset,
    auto_detect_baseline_start,
)
from .model import DSCPeakBot, train_model, predict_on_curve
from .adaptive_boundary import BOUNDARY_POLICIES, find_adaptive_boundaries
from .thermal_metrics import calculate_extrapolated_onset_endset
from .main import run_training, run_detection, export_results
from .visualization import plot_detection, plot_all_detections
