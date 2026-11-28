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
from .adaptive_boundary import BOUNDARY_POLICIES, find_adaptive_boundaries
from .thermal_metrics import calculate_extrapolated_onset_endset

# Data preparation, configuration checks and routing tests do not need
# TensorFlow or the plotting backend. Keep the established public names,
# importing these optional modules only when the corresponding API is used.
_LAZY_EXPORTS = {
    "DSCPeakBot": (".model", "DSCPeakBot"),
    "train_model": (".model", "train_model"),
    "predict_on_curve": (".model", "predict_on_curve"),
    "run_training": (".main", "run_training"),
    "run_detection": (".main", "run_detection"),
    "export_results": (".main", "export_results"),
    "plot_detection": (".visualization", "plot_detection"),
    "plot_all_detections": (".visualization", "plot_all_detections"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    from importlib import import_module
    module_name, attribute = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
