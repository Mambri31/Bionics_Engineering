"""
LabVIEW Inference Script
========================
This script is designed to be called from LabVIEW's Python Node
on every iteration of the block diagram.

It loads the trained model (.pkl) specified by `model_path` and returns
the predicted gesture class as a native Python int.

Expected inputs:
    data_array : list/array of doubles (generic — spike counts, distances, etc.)
    model_path : str — absolute path to the .pkl model file, supplied by LabVIEW

Expected output: int (predicted gesture class, 1-6)

Gesture classes:
    1 - Open palm
    2 - Fist
    3 - Thumb only
    4 - Pinky only
    5 - Thumb + pinky
    6 - Index + ring + thumb (horns sign)
"""

import numpy as np
import joblib

# ---------------------------------------------------------------------------
# Cache: the model is loaded only once per (path) and reused for every
# subsequent call within the same LabVIEW session.
# If the path changes at runtime, the new model is loaded automatically.
# ---------------------------------------------------------------------------
_model = None
_loaded_path = None


def predict_gesture(data_array, model_path):
    """
    Predict the gesture class from a generic numeric array.

    Parameters
    ----------
    data_array : list or array-like
        Generic input array of doubles (spike counts, distances, etc.).
        Its length must match the number of features expected by the model.
    model_path : str
        Absolute path to the serialized model file (.pkl),
        provided directly by LabVIEW.

    Returns
    -------
    int
        Predicted gesture class (native Python int, LabVIEW-compatible).
    """
    global _model, _loaded_path

    # Lazy-load the model on first call, or reload if the path changed
    if _model is None or _loaded_path != model_path:
        _model = joblib.load(model_path)
        _loaded_path = model_path

    # Reshape to (1, N) — single sample with N features
    features = np.array(data_array, dtype=np.float64).reshape(1, -1)

    # Predict and cast to native Python int (LabVIEW expects int, not numpy.int64)
    prediction = _model.predict(features)
    return int(prediction[0])
