import threading
import os
import numpy as np
import tensorflow as tf

_MODEL = None
_LOCK = threading.Lock()

def _force_build_model(m):
    """
    Force the model to build all its layers by doing forward passes.
    This is critical for models with Sequential layers that haven't been called.
    """
    print("[INFO] Forcing model build...")
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    
    try:
        # Call the main model
        _ = m(dummy, training=False)
        print("[OK] Main model called successfully")
        
        # If first layer is Sequential, call it explicitly
        if len(m.layers) > 0 and isinstance(m.layers[0], tf.keras.Sequential):
            print("[INFO] Found Sequential layer, calling it explicitly...")
            _ = m.layers[0](dummy, training=False)
            print("[OK] Sequential layer built")
        
        # Verify all layers are built
        print("[INFO] Model architecture:")
        for i, layer in enumerate(m.layers):
            print(f"  Layer {i}: {layer.name} - {type(layer).__name__}")
            if hasattr(layer, 'layers'):
                for j, sublayer in enumerate(layer.layers):
                    print(f"    Sublayer {j}: {sublayer.name} - {type(sublayer).__name__}")
        
    except Exception as e:
        print(f"[ERROR] Failed to build model: {e}")
        raise
    
    print("[OK] Model fully initialized")

def get_model(model_filename="cnn_fake_detector.h5"):
    """
    Load and return the deepfake detection model (singleton pattern).
    
    Args:
        model_filename: Name of the H5 model file
    
    Returns:
        Loaded and compiled Keras model
    """
    global _MODEL
    
    # Return cached model if available
    if _MODEL is not None:
        return _MODEL
    
    # Construct absolute path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, model_filename)
    
    # Check file exists
    if not os.path.exists(model_path):
        # Try alternative path
        alt_path = os.path.join(os.getcwd(), model_filename)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(
                f"Model not found at {model_path} or {alt_path}. "
                f"Please ensure {model_filename} is in the project root."
            )
    
    # Thread-safe loading
    with _LOCK:
        if _MODEL is None:
            print(f"[INFO] Loading model from: {model_path}")
            
            try:
                # Load without compilation
                _MODEL = tf.keras.models.load_model(
                    model_path, 
                    compile=False
                )
                print("[OK] Model loaded successfully")
                
                # Compile the model
                _MODEL.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                print("[OK] Model compiled")
                
                # Force build all layers
                _force_build_model(_MODEL)
                
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
                raise
    
    return _MODEL

def reload_model():
    """
    Force reload the model (useful for debugging).
    """
    global _MODEL
    with _LOCK:
        _MODEL = None
    return get_model()

def get_model_info():
    """
    Get information about the loaded model.
    
    Returns:
        dict: Model information including layers, input shape, etc.
    """
    model = get_model()
    
    info = {
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'total_params': model.count_params(),
        'layers': []
    }
    
    for layer in model.layers:
        layer_info = {
            'name': layer.name,
            'type': type(layer).__name__,
            'output_shape': layer.output_shape
        }
        info['layers'].append(layer_info)
    
    return info
