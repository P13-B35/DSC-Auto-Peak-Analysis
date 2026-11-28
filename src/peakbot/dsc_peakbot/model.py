"""1D CNN model for DSC melt-peak detection."""

import os
import uuid
import pickle

import numpy as np
import tensorflow as tf

from .config import Config
from .core import tic, toc, timeit


# ---------------------------------------------------------------------------
# Custom metrics
# ---------------------------------------------------------------------------

def _iou_1d(boxes_true, boxes_pred):
    """Intersection-over-Union for 1D bounding boxes.

    Args:
        boxes_true: (batch, 2) — [left, right].
        boxes_pred: (batch, 2) — [left, right].

    Returns:
        IoU per sample, shape (batch,).
    """
    left  = tf.maximum(boxes_true[:, 0], boxes_pred[:, 0])
    right = tf.minimum(boxes_true[:, 1], boxes_pred[:, 1])
    intersection = tf.maximum(right - left, 0)

    len_true  = boxes_true[:, 1] - boxes_true[:, 0]
    len_pred  = boxes_pred[:, 1] - boxes_pred[:, 0]
    union = len_true + len_pred - intersection + tf.constant(1e-8)

    return intersection / union


def iou_metric(y_true, y_pred):
    """Keras metric wrapper for box IoU."""
    return tf.reduce_mean(_iou_1d(y_true, y_pred))


def _recall(y_true, y_pred):
    tp = tf.keras.backend.sum(
        tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    pp = tf.keras.backend.sum(
        tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    return tp / (pp + tf.keras.backend.epsilon())


def _precision(y_true, y_pred):
    tp = tf.keras.backend.sum(
        tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    pred_pos = tf.keras.backend.sum(
        tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    return tp / (pred_pos + tf.keras.backend.epsilon())


def _f1(y_true, y_pred):
    p = _precision(y_true, y_pred)
    r = _recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))


def pF1(y_true, y_pred):
    """F1 score for 'is-peak vs not-peak' (binary reduction)."""
    yt = tf.cast(tf.math.less(tf.math.argmax(y_true, axis=1),
                               Config.IS_PEAK_CLASSES), tf.float32)
    yp = tf.cast(tf.math.less(tf.math.argmax(y_pred, axis=1),
                               Config.IS_PEAK_CLASSES), tf.float32)
    return _f1(yt, yp)


def pTPR(y_true, y_pred):
    """True-positive rate for peak detection."""
    yt = tf.cast(tf.math.less(tf.math.argmax(y_true, axis=1),
                               Config.IS_PEAK_CLASSES), tf.float32)
    yp = tf.cast(tf.math.less(tf.math.argmax(y_pred, axis=1),
                               Config.IS_PEAK_CLASSES), tf.float32)
    return _recall(yt, yp)


def pFPR(y_true, y_pred):
    """False-positive rate for peak detection."""
    yt = tf.cast(tf.math.less(tf.math.argmax(y_true, axis=1),
                               Config.IS_PEAK_CLASSES), tf.float32)
    yp = tf.cast(tf.math.less(tf.math.argmax(y_pred, axis=1),
                               Config.IS_PEAK_CLASSES), tf.float32)
    return 1.0 - _precision(yt, yp)


# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------

def data_generator(instances_folder, prefix=None, verbose=False):
    """Yield batches of training instances from a folder of pickle files.

    Yields:
        (inputs_dict, targets_dict)
    """
    if prefix is None:
        prefix = Config.INSTANCE_PREFIX

    idx = 0
    while True:
        fpath = os.path.join(instances_folder, f"{prefix}{idx}.pickle")
        if not os.path.isfile(fpath):
            if idx == 0:
                raise FileNotFoundError(
                    f"No instance files found in '{instances_folder}' "
                    f"with prefix '{prefix}'")
            idx = 0
            continue

        batch = pickle.load(open(fpath, 'rb'))
        area     = batch['area']       # (batch, window_size)
        feats    = batch['features']   # (batch, 3)
        ptype    = batch['type']       # (batch, num_classes)
        center   = batch['center']     # (batch, 1)
        box      = batch['box']        # (batch, 2)

        # Add channel dimension for Conv1D
        area = area[..., np.newaxis]  # (batch, window_size, 1)

        yield (
            {'area': area, 'features': feats},
            {'peak_type': ptype,
             'center':    center,
             'box':       box}
        )
        idx += 1


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class DSCPeakBot:
    """1D CNN model for DSC melt-peak detection."""

    def __init__(self, name=None):
        self.name    = name or f"DSCPeakBot_v{Config.VERSION}_{uuid.uuid4().hex[:6]}"
        self.model   = None
        self.window_size = Config.WINDOW_SIZE
        self.num_classes = Config.NUM_CLASSES

    @timeit
    def build_model(self, verbose=False):
        """Construct the 1D CNN architecture with dual inputs.

        Input 1:  area     — (window_size, 1) normalised heat-flow window.
        Input 2:  features — (3,) dimensionless strength/SNR features that
                  complement the normalised local shape.

        Output: peak_type (num_classes), center (1), box (2)
        """
        layer_sizes = Config.CONV_LAYER_SIZES
        dropout     = Config.DROPOUT
        n_features  = 3

        area = tf.keras.Input(shape=(self.window_size, 1), name='area')
        feats = tf.keras.Input(shape=(n_features,), name='features')

        if verbose:
            print(f"  | Building model '{self.name}'")
            print(f"  | .. area input: {area}")
            print(f"  | .. features input: {feats}")

        # CNN branch
        x = area
        for i, size in enumerate(layer_sizes):
            x = tf.keras.layers.Conv1D(size, 5, padding='same',
                                        use_bias=False, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization(scale=False)(x)
            x = tf.keras.layers.Conv1D(size, 3, padding='same',
                                        use_bias=False, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization(scale=False)(x)
            x = tf.keras.layers.MaxPooling1D(2)(x)
            x = tf.keras.layers.Dropout(dropout)(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Flatten()(x)

        # Scalar-feature branch — light MLP to project features into a
        # comparable dimensionality before concatenation.
        f = tf.keras.layers.Dense(8, activation='relu')(feats)
        f = tf.keras.layers.BatchNormalization()(f)

        # Fuse
        x = tf.keras.layers.Concatenate()([x, f])

        # Outputs
        peak_type = tf.keras.layers.Dense(self.num_classes, name='peak_type',
                                           activation='softmax')(x)
        center    = tf.keras.layers.Dense(1, name='center',
                                           activation='relu')(x)
        box       = tf.keras.layers.Dense(2, name='box',
                                           activation='relu')(x)

        if verbose:
            print(f"  | .. peak_type: {peak_type}")
            print(f"  | .. center:    {center}")
            print(f"  | .. box:       {box}")

        self.model = tf.keras.models.Model([area, feats], [peak_type, center, box])
        return self

    @timeit
    def compile_model(self, learning_rate=None):
        """Compile the model with loss functions and metrics."""
        if learning_rate is None:
            learning_rate = Config.LEARNING_RATE_START

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'peak_type': 'categorical_crossentropy',
                'center':    tf.keras.losses.Huber(),
                'box':       tf.keras.losses.Huber(),
            },
            metrics={
                'peak_type': ['categorical_accuracy', pF1, pTPR, pFPR],
                'center':    [],
                'box':       [iou_metric],
            },
            loss_weights={
                'peak_type': 1.0,
                'center':    1.0 / 25.0,
                'box':       1.0 / 25.0,
            },
        )
        return self

    @timeit
    def train(self, train_generator, val_data=None, log_dir=None,
              callbacks=None, verbose=1, tensorboard=True):
        """Train the model.

        Args:
            train_generator: Generator yielding (inputs, targets) tuples.
            val_data: Optional validation data in the same format.
            log_dir: TensorBoard log directory.
            callbacks: Additional Keras callbacks.
            verbose: Keras verbosity level.
            tensorboard: Enable TensorBoard files (disable on unsupported paths).
        """
        epochs = Config.EPOCHS
        steps  = Config.STEPS_PER_EPOCH

        if log_dir is None:
            log_dir = os.path.join(
                'logs', 'fit',
                f"DSCPeakBot_v{Config.VERSION}_{uuid.uuid4().hex}")

        if tensorboard:
            os.makedirs(log_dir, exist_ok=True)

        if verbose:
            print(f"  | Training model '{self.name}'")
            print(f"  | .. epochs: {epochs}, steps/epoch: {steps}")
            print(f"  | .. log dir: {log_dir}")

        _callbacks = []
        if tensorboard and os.path.isdir(log_dir):
            tensorboard_cb = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=0, write_graph=False)
            _callbacks = [tensorboard_cb]
        elif verbose:
            print("  | .. TensorBoard log dir not available, skipping")
        if callbacks is not None:
            _callbacks.extend(
                callbacks if isinstance(callbacks, list) else [callbacks])

        history = self.model.fit(
            train_generator,
            validation_data=val_data,
            batch_size=Config.BATCH_SIZE,
            epochs=epochs,
            steps_per_epoch=steps,
            callbacks=_callbacks,
            verbose=verbose,
        )
        return history

    def predict(self, areas, features=None):
        """Run prediction on a batch of windows.

        Args:
            areas: (N, window_size) or (N, window_size, 1) array.
            features: (N, 3) scalar features array, or None.

        Returns:
            (peak_type, center, box) — each a numpy array.
        """
        if areas.ndim == 2:
            areas = areas[..., np.newaxis]
        if features is not None:
            return self.model.predict({'area': areas, 'features': features},
                                       verbose=0)
        return self.model.predict(areas, verbose=0)

    def save(self, filepath):
        """Save the model to disk."""
        self.model.save(filepath)

    def load(self, filepath):
        """Load a saved model."""
        custom_objects = {
            'iou_metric': iou_metric, 'pF1': pF1,
            'pTPR': pTPR, 'pFPR': pFPR,
        }
        # h5py supports Windows Unicode paths that TensorFlow 2.13's
        # directory probing cannot decode. Passing the open HDF5 object
        # avoids that filesystem probe without copying or altering weights.
        if os.path.isfile(filepath) and str(filepath).lower().endswith(('.h5', '.hdf5')):
            import h5py
            with h5py.File(filepath, 'r') as handle:
                self.model = tf.keras.models.load_model(
                    handle, custom_objects=custom_objects)
        else:
            self.model = tf.keras.models.load_model(
                filepath, custom_objects=custom_objects)


# ---------------------------------------------------------------------------
# Top-level training / prediction helpers
# ---------------------------------------------------------------------------

@timeit
def train_model(instances_folder, model_output_path,
                val_instances_folder=None,
                log_base_dir=None, verbose=True):
    """Train a new DSCPeakBot model from prepared instance files.

    Args:
        instances_folder: Folder of training pickle files.
        model_output_path: Where to save the trained model (.h5 or SavedModel).
        val_instances_folder: Optional folder of validation pickle files.
        log_base_dir: Base directory for TensorBoard logs.
        verbose: Print progress.
    """
    if verbose:
        print("Training new DSC PeakBot model")
        print(Config.get_as_string())

    # Learning rate schedule
    def lr_schedule(epoch, lr):
        if (epoch + 1) % Config.LEARNING_RATE_DECREASE_AFTER == 0:
            lr = float(lr) * Config.LEARNING_RATE_MULTIPLIER
        tf.summary.scalar('learning_rate', data=lr, step=epoch)
        return max(lr, Config.LEARNING_RATE_MIN)

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lr_schedule, verbose=0)

    # Resolve log base to an absolute path — avoid Unicode paths that
    # can cause TensorBoard WriteFile errors on Windows.
    if log_base_dir is None:
        log_base_dir = os.path.join(
            os.environ.get('TEMP', os.path.expanduser('~')),
            'dsc_peakbot_logs')
    os.makedirs(log_base_dir, exist_ok=True)

    csv_log_path = os.path.join(log_base_dir,
                                 f"training_{uuid.uuid4().hex[:6]}.tsv")
    csv_logger = tf.keras.callbacks.CSVLogger(csv_log_path, separator='\t')

    train_gen = data_generator(instances_folder, verbose=verbose)
    val_gen   = None
    if val_instances_folder is not None:
        # Take one batch as fixed validation set
        val_gen = data_generator(val_instances_folder, verbose=verbose)
        val_x, val_y = next(val_gen)
        val_gen = (val_x, val_y)  # fixed — won't regenerate each epoch

    log_dir = os.path.join(log_base_dir,
                            f"DSCPeakBot_{uuid.uuid4().hex[:6]}")
    os.makedirs(log_dir, exist_ok=True)

    pb = DSCPeakBot()
    pb.build_model(verbose=verbose)
    pb.compile_model()

    pb.train(train_gen,
             val_data=val_gen,
             log_dir=log_dir,
             callbacks=[lr_scheduler, csv_logger],
             verbose=2 if verbose else 0)

    pb.save(model_output_path)
    if verbose:
        print(f"  | Model saved to '{model_output_path}'")

    return pb


@timeit
def predict_on_curve(pb, areas, features, temp_win, gd_props, verbose=True):
    """Run prediction on instances from a single DSC curve.

    Args:
        pb: DSCPeakBot instance (trained model).
        areas: (N, window_size) array.
        features: (N, 3) scalar features array.
        temp_win: (N, window_size) temperature windows.
        gd_props: (N, 7) boundary + signal info [ct_temp, ct_hf, left_temp, right_temp, apex, full_depth, full_SNR].

    Returns:
        peaks: list of [center_temp, center_hf, temp_start, temp_end, type_idx].
    """
    n = areas.shape[0]
    if n == 0:
        return []

    # Predict in batches to avoid OOM
    batch_size = Config.BATCH_SIZE * 4
    peak_type_all = []
    center_all    = []
    box_all       = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        pt, ct, bx = pb.predict(areas[start:end],
                                 features=features[start:end])
        peak_type_all.append(pt)
        center_all.append(ct)
        box_all.append(bx)

    ptype  = np.concatenate(peak_type_all, axis=0)  # (N, num_classes)
    center = np.concatenate(center_all, axis=0)      # (N, 1)
    box    = np.concatenate(box_all, axis=0)         # (N, 2)

    peaks = []
    for i in range(n):
        cls_idx = int(np.argmax(ptype[i]))
        ct_idx  = int(round(center[i, 0]))
        left    = int(round(box[i, 0]))
        right   = int(round(box[i, 1]))

        win_sz = temp_win.shape[1]

        # Centre regression is retained as auxiliary supervision; final
        # localisation uses the actual full-curve extremum candidate.
        ct_temp   = gd_props[i, 0]
        ct_hf     = gd_props[i, 1]
        # Use full-curve boundary detection (gd_props) rather than
        # CNN box regression, which is limited by the 128-pt window.
        left_temp = gd_props[i, 2]
        right_temp = gd_props[i, 3]

        peak_depth = float(gd_props[i, 5])
        peak_snr = float(gd_props[i, 6])
        span = float(right_temp - left_temp)
        rel_pos = None
        if span > 1e-6:
            rel_pos = float((ct_temp - left_temp) / span)

        # Unit-invariant SNR filter; absolute uV/mW thresholds are avoided.
        if cls_idx < Config.IS_PEAK_CLASSES and peak_snr < Config.MIN_PEAK_SNR:
            cls_idx = 2  # re-classify as artifact

        # Boundary location is used for quantification, not to overturn
        # the CNN class.  This decoupling prevents a noisy shoulder estimate
        # from suppressing an otherwise correctly recognised broad peak.

        peaks.append({
            'center_temp': float(ct_temp),
            'center_hf':   float(ct_hf),
            'temp_start':  float(left_temp),
            'temp_end':    float(right_temp),
            'type_idx':    cls_idx,
            'type_name':   {0: 'melt_peak', 1: 'baseline', 2: 'artifact'}[cls_idx],
            'full_depth':  peak_depth,
            'full_snr':    peak_snr,
            'confidence':  float(ptype[i, cls_idx]),
        })

    if verbose:
        n_peaks = sum(1 for p in peaks if p['type_idx'] < Config.IS_PEAK_CLASSES)
        print(f"  | Detected {n_peaks} melt peaks out of {n} local maxima "
              f"in curve")

    return peaks
