import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array
from utils.model_utils import get_model
from utils.xai_utils import (
    make_gradcam_heatmap, 
    overlay_heatmap_on_image,
    create_explanation_visualization
)

def preprocess_image(img):
    """
    Preprocess image for model input.
    
    Args:
        img: BGR image from OpenCV
    
    Returns:
        img_array: Preprocessed image array ready for model
    """
    img_resized = cv2.resize(img, (224, 224))
    img_array = img_to_array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

def predict_image(img, model=None):
    """
    Predict if an image is real or fake.
    
    Args:
        img: Input BGR image
        model: Keras model (loads if None)
    
    Returns:
        float: Prediction score (0=Real, 1=Fake)
    """
    if model is None:
        model = get_model()
    
    img_batch = preprocess_image(img)
    pred = model.predict(img_batch, verbose=0)[0][0]
    return float(pred)

def predict_image_with_xai(img, model=None, save_path=None):
    """
    Predict image with XAI explanation.
    
    Args:
        img: Input BGR image or file path
        model: Keras model
        save_path: Optional path to save explanation
    
    Returns:
        tuple: (prediction, label, explanation_image)
    """
    # Load image if path provided
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise ValueError(f"Could not load image from path")
    
    if model is None:
        model = get_model()
    
    # Get prediction
    img_batch = preprocess_image(img)
    pred = model.predict(img_batch, verbose=0)[0][0]
    label = "Fake" if pred > 0.5 else "Real"
    
    # Generate heatmap
    try:
        heatmap = make_gradcam_heatmap(model, img_batch)
        explanation_img = create_explanation_visualization(img, heatmap, pred, label)
        
        if save_path:
            cv2.imwrite(save_path, explanation_img)
            print(f"[OK] Explanation saved to {save_path}")
            
    except Exception as e:
        print(f"[WARNING] Could not generate explanation: {e}")
        # Fallback to simple text annotation
        img_resized = cv2.resize(img, (224, 224))
        confidence = pred * 100 if pred > 0.5 else (1 - pred) * 100
        color = (0, 0, 255) if pred > 0.5 else (0, 255, 0)
        text = f"{label} ({confidence:.1f}%)"
        explanation_img = img_resized.copy()
        cv2.putText(explanation_img, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return pred, label, explanation_img

def predict_video(video_path, frame_interval=10, threshold=0.5):
    """
    Analyze video for deepfake detection.
    
    Args:
        video_path: Path to video file
        frame_interval: Analyze every Nth frame
        threshold: Classification threshold
    
    Returns:
        tuple: (result, avg_score, frame_details)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    model = get_model()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_preds = []
    frame_details = []
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[INFO] Analyzing video: {total_frames} frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            try:
                pred = predict_image(frame, model)
                frame_preds.append(pred)
                frame_details.append((frame_idx, float(pred)))
                
                if frame_idx % 100 == 0:
                    print(f"[INFO] Processed frame {frame_idx}/{total_frames}")
                    
            except Exception as e:
                print(f"[WARNING] Error processing frame {frame_idx}: {e}")
        
        frame_idx += 1
    
    cap.release()
    
    if len(frame_preds) == 0:
        return "Error: No frames processed", 0.0, []
    
    avg_score = np.mean(frame_preds)
    result = "Fake" if avg_score > threshold else "Real"
    
    print(f"[RESULT] Video classified as: {result} (score: {avg_score:.4f})")
    
    return result, float(avg_score), frame_details

def analyze_video_with_xai(video_path, output_path=None, frame_interval=10, 
                          threshold=0.5, show_heatmap=True):
    """
    Analyze video and create annotated output with XAI visualizations.
    
    Args:
        video_path: Input video path
        output_path: Output video path (optional)
        frame_interval: Process every Nth frame
        threshold: Classification threshold
        show_heatmap: Show Grad-CAM heatmap overlay
    
    Returns:
        tuple: (result, avg_score, output_path)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    model = get_model()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"[INFO] Writing output to: {output_path}")
    
    frame_preds = []
    frame_idx = 0
    
    print(f"[INFO] Processing {total_frames} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process at intervals
        if frame_idx % frame_interval == 0:
            try:
                img_batch = preprocess_image(frame)
                pred = model.predict(img_batch, verbose=0)[0][0]
                frame_preds.append(pred)
                
                # Add visualization
                if show_heatmap:
                    try:
                        heatmap = make_gradcam_heatmap(model, img_batch)
                        heatmap_resized = cv2.resize(heatmap, (width, height))
                        frame = overlay_heatmap_on_image(frame, heatmap_resized, alpha=0.3)
                    except Exception as e:
                        print(f"[WARNING] Heatmap failed for frame {frame_idx}: {e}")
                
                # Add text
                label = "FAKE" if pred > threshold else "REAL"
                color = (0, 0, 255) if pred > threshold else (0, 255, 0)
                cv2.putText(frame, f"{label} ({pred:.2%})", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                cv2.putText(frame, f"Frame: {frame_idx}", (20, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
            except Exception as e:
                print(f"[ERROR] Frame {frame_idx} processing failed: {e}")
        
        if writer:
            writer.write(frame)
        
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            print(f"[INFO] Progress: {frame_idx}/{total_frames} frames")
    
    cap.release()
    if writer:
        writer.release()
        print(f"[OK] Output video saved: {output_path}")
    
    if len(frame_preds) == 0:
        return "Error: No frames processed", 0.0, output_path
    
    avg_score = np.mean(frame_preds)
    result = "Fake" if avg_score > threshold else "Real"
    
    print(f"[RESULT] Video: {result} (avg score: {avg_score:.4f})")
    
    return result, float(avg_score), output_path

def extract_keyframes_with_xai(video_path, num_keyframes=10, output_dir=None):
    """
    Extract keyframes from video and analyze with XAI.
    
    Args:
        video_path: Input video path
        num_keyframes: Number of keyframes to extract
        output_dir: Directory to save keyframe explanations
    
    Returns:
        tuple: (overall_result, avg_score, keyframe_data)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    model = get_model()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_keyframes, dtype=int)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    keyframe_data = []
    scores = []
    
    print(f"[INFO] Extracting {num_keyframes} keyframes...")
    
    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"[WARNING] Could not read frame {frame_idx}")
            continue
        
        try:
            # Get prediction with explanation
            pred, label, explanation_img = predict_image_with_xai(frame, model)
            scores.append(pred)
            
            # Save if output directory specified
            if output_dir:
                filename = f"keyframe_{idx:03d}_frame_{frame_idx:06d}.jpg"
                save_path = os.path.join(output_dir, filename)
                cv2.imwrite(save_path, explanation_img)
            
            keyframe_data.append({
                'index': idx,
                'frame_number': int(frame_idx),
                'prediction': float(pred),
                'label': label
            })
            
            print(f"[INFO] Keyframe {idx+1}/{num_keyframes}: {label} ({pred:.4f})")
            
        except Exception as e:
            print(f"[ERROR] Failed to process keyframe {frame_idx}: {e}")
    
    cap.release()
    
    if len(scores) == 0:
        return "Error: No keyframes processed", 0.0, []
    
    avg_score = np.mean(scores)
    result = "Fake" if avg_score > 0.5 else "Real"
    
    print(f"[RESULT] Overall: {result} (avg: {avg_score:.4f})")
    
    return result, float(avg_score), keyframe_data
