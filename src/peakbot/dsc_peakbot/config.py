"""DSC PeakBot configuration."""

# Standard CNN input length and target temperature span in degrees Celsius.
PAPER_WINDOWS = {"narrow": (120, 2.0), "broad": (240, 4.0)}


class Config:
    """Configuration for DSC PeakBot — 1D CNN peak detection."""

    NAME    = "DSC_PeakBot"
    VERSION = "1.0"

    # 1D window size extracted around each local maximum
    WINDOW_SIZE = 128

    # Minimum temperature for peak detection (exclude preparation phase)
    MIN_TEMPERATURE = 80.0  # °C

    # Unit-invariant post-filter. Development data are in uV and final test
    # data are in mW, so an absolute amplitude threshold would be invalid.
    MIN_PEAK_SNR = 4.0

    # Boundary detection — threshold-crossing fraction of peak depth
    PEAK_FRACTION = 0.90  # stop when signal returns to 90% of baseline
    SG_BOUNDARY   = 31    # Savitzky-Golay window for boundary smoothing

    # [melt_peak, baseline, artifact]
    NUM_CLASSES      = 3
    IS_PEAK_CLASSES  = 1   # first N classes represent a real peak

    # CNN architecture
    BATCH_SIZE       = 32
    STEPS_PER_EPOCH  = 64
    EPOCHS           = 256

    DROPOUT           = 0.2
    CONV_LAYER_SIZES = [32, 64, 128, 256]

    # Learning rate schedule
    LEARNING_RATE_START              = 0.0005
    LEARNING_RATE_DECREASE_AFTER     = 5
    LEARNING_RATE_MULTIPLIER         = 0.96
    LEARNING_RATE_MIN                = 3e-7

    # Baseline construction — iterative polynomial fitting
    BASELINE_POLY_DEGREE   = 3      # polynomial degree
    BASELINE_MAX_ITER       = 10     # max iterations
    BASELINE_CONVERGENCE    = 1e-4   # stop when baseline change < this (mW)
    BASELINE_SIGMA          = 3.0    # n_std threshold to flag peak regions

    # Enthalpy calculation
    DEFAULT_HEATING_RATE   = 10.0    # °C/min (typical DSC)
    # sample mass (mg) must be provided by user — no default

    # Training instance generation
    INSTANCE_PREFIX = "___DSC_PB_sample_"

    @staticmethod
    def get_as_string():
        lines = [
            f"  | .. {Config.NAME}",
            f"  | .. Version {Config.VERSION}",
            f"  | .. Window size: {Config.WINDOW_SIZE}",
            f"  | .. Number of classes: {Config.NUM_CLASSES}",
            f"  | .. Peak classes: {Config.IS_PEAK_CLASSES}",
            f"  | .. Batch size: {Config.BATCH_SIZE}, Epochs: {Config.EPOCHS}",
            f"  | .. Conv layer sizes: {Config.CONV_LAYER_SIZES}",
            f"  | .. Dropout rate: {Config.DROPOUT}",
            f"  | .. Learning rate: start={Config.LEARNING_RATE_START}, "
            f"decay after {Config.LEARNING_RATE_DECREASE_AFTER} steps, "
            f"multiplier={Config.LEARNING_RATE_MULTIPLIER}",
        ]
        return "\n".join(lines)
