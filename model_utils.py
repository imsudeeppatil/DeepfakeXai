import threading
import os
import numpy as np
import tensorflow as tf

_MODEL = None
_LOCK = threading.Lock()

def _force_build_model(m):
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    _ = m(dummy, training=False)
    if isinstance(m.layers[0], tf.keras.Sequential):
        _ = m.layers[0](dummy, training=False)

def get_model(model_filename="C:\Major Project Backup\cnn_fake_detector.h5"):
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Model not found: {model_path}")

    with _LOCK:
        if _MODEL is None:
            _MODEL = tf.keras.models.load_model(model_path, compile=False)
            try:
                _MODEL.compile(optimizer="adam", loss="binary_crossentropy")
            except Exception:
                pass

            _force_build_model(_MODEL)
            print("[OK] Model loaded and built successfully:", model_path)

    return _MODEL
