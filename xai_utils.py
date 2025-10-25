import numpy as np
import cv2
import tensorflow as tf

def find_last_conv_layer(model):
    """
    Find the last convolutional layer in the model.
    Handles both direct Conv2D layers and nested Sequential models.
    """
    # Check direct layers first
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    
    # Check nested layers (Sequential, Functional)
    for layer in model.layers:
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    
    raise ValueError("No Conv2D layer found in the model.")


def make_gradcam_heatmap(model, img_array, last_conv_layer_name=None, pred_index=None):
    """
    Generate Grad-CAM heatmap for an input image.
    
    Args:
        model: Trained Keras model
        img_array: Preprocessed image array (batch_size, height, width, channels)
        last_conv_layer_name: Name of target conv layer (auto-detected if None)
        pred_index: Target class index (None for binary classification)
    
    Returns:
        heatmap_resized: Heatmap resized to input dimensions
    """
    # Find target layer if not specified
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)
        print(f"[INFO] Using convolutional layer: {last_conv_layer_name}")
    
    # Ensure model is built by doing a forward pass first
    try:
        _ = model.predict(img_array, verbose=0)
    except Exception as e:
        print(f"[WARNING] Initial prediction failed: {e}")
    
    # Get the conv layer
    try:
        conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError as e:
        print(f"[ERROR] Could not find layer {last_conv_layer_name}: {e}")
        raise
    
    # Create gradient model
    try:
        grad_model = tf.keras.models.Model(
            inputs=[model.input],
            outputs=[conv_layer.output, model.output]
        )
    except Exception as e:
        print(f"[ERROR] Could not create gradient model: {e}")
        raise
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        
        # Handle binary classification
        if pred_index is None:
            if predictions.shape[-1] == 1:
                # Single output neuron
                loss = predictions[:, 0]
            else:
                # Multiple outputs - use argmax
                pred_index = int(tf.argmax(predictions[0]))
                loss = predictions[:, pred_index]
        else:
            loss = predictions[:, pred_index]
    
    # Calculate gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Global average pooling on gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the feature maps
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    
    # Create heatmap
    heatmap = np.mean(conv_outputs, axis=-1)
    
    # Apply ReLU and normalize
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    
    # Resize to match input image
    h, w = img_array.shape[1], img_array.shape[2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    return heatmap_resized


def overlay_heatmap_on_image(original_bgr, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        original_bgr: Original image in BGR format
        heatmap: Grad-CAM heatmap (normalized 0-1)
        alpha: Transparency factor for heatmap
        colormap: OpenCV colormap to apply
    
    Returns:
        overlay: Blended image with heatmap
    """
    # Resize heatmap if needed
    if heatmap.shape != original_bgr.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_bgr.shape[1], original_bgr.shape[0]))
    
    # Convert heatmap to uint8
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Blend with original
    overlay = cv2.addWeighted(heatmap_colored, alpha, original_bgr, 1 - alpha, 0)
    
    return overlay


def create_explanation_visualization(original_img, heatmap, prediction, label):
    """
    Create a comprehensive visualization with original, heatmap, and overlay.
    
    Args:
        original_img: Original BGR image
        heatmap: Grad-CAM heatmap
        prediction: Raw prediction score (0-1)
        label: Classification label ("Real" or "Fake")
    
    Returns:
        visualization: Combined visualization image
    """
    h, w = original_img.shape[:2]
    
    # Resize for consistent display
    display_size = (300, 300)
    original_resized = cv2.resize(original_img, display_size)
    
    # Create heatmap visualization
    heatmap_resized = cv2.resize(heatmap, display_size)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Create overlay
    overlay = overlay_heatmap_on_image(original_resized, heatmap_resized, alpha=0.5)
    
    # Stack horizontally
    combined = np.hstack([original_resized, heatmap_colored, overlay])
    
    # Add labels
    labels = ["Original", "Attention Map", "Overlay"]
    for i, text in enumerate(labels):
        x_pos = i * display_size[0] + 10
        cv2.putText(combined, text, (x_pos, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, text, (x_pos, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Add prediction info at bottom
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
    pred_text = f"Prediction: {label} ({confidence:.1f}% confidence)"
    color = (0, 0, 255) if label == "Fake" else (0, 255, 0)
    
    # Create info bar
    info_bar = np.zeros((60, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(info_bar, pred_text, (10, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Combine
    visualization = np.vstack([combined, info_bar])
    
    return visualization
