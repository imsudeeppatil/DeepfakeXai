import sys
import io

# Force UTF-8 encoding for console output (fixes emoji issues)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from flask import Flask, request, jsonify, send_from_directory, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
KEYFRAMES_FOLDER = 'static/keyframes'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'}
MODEL_PATH = r'c:\Major Project Backup\cnn_fake_detector.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['KEYFRAMES_FOLDER'] = KEYFRAMES_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# Create folders if they don't exist
print("[INFO] Checking/creating required folders...")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(KEYFRAMES_FOLDER, exist_ok=True)
print(f"[OK] Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
print(f"[OK] Output folder: {os.path.abspath(OUTPUT_FOLDER)}")
print(f"[OK] Keyframes folder: {os.path.abspath(KEYFRAMES_FOLDER)}")

# Global model variable
model = None
TARGET_SIZE = (256, 256)  # Will be auto-detected during model loading

def convert_to_python_type(obj):
    """Convert NumPy types to Python native types recursively"""
    if isinstance(obj, dict):
        return {key: convert_to_python_type(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_type(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def allowed_file(filename, file_type='image'):
    """Check if file extension is allowed"""
    if file_type == 'image':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    elif file_type == 'video':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS
    return False

def load_model():
    """Load the pre-trained model and detect correct input size"""
    global model, TARGET_SIZE
    try:
        print("[INFO] Pre-loading model...")
        print(f"[INFO] Loading model from: {MODEL_PATH}")
        
        model = keras.models.load_model(MODEL_PATH)
        print("[OK] Model loaded successfully")
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("[OK] Model compiled")
        
        # Detect correct input size
        print("[INFO] Detecting correct input size...")
        test_sizes = [256, 224, 128, 299, 384, 512]
        correct_size = None
        
        for size in test_sizes:
            try:
                print(f"[INFO] Testing size {size}x{size}...")
                dummy_input = np.zeros((1, size, size, 3), dtype=np.float32)
                _ = model.predict(dummy_input, verbose=0)
                correct_size = size
                print(f"[OK] Input size {size}x{size} works!")
                break
            except Exception as e:
                print(f"[SKIP] Size {size}x{size} failed")
                continue
        
        if correct_size is None:
            raise ValueError("Could not detect correct input size. Please check your model.")
        
        # Update target size
        TARGET_SIZE = (correct_size, correct_size)
        print(f"[OK] Using input size: {TARGET_SIZE}")
        
        # Print model architecture
        print("[INFO] Model architecture:")
        for i, layer in enumerate(model.layers):
            print(f"  Layer {i}: {layer.name} - {layer.__class__.__name__}")
        
        print("[OK] Model fully initialized")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def preprocess_image(image_path, target_size=None):
    """Preprocess image for model prediction"""
    try:
        if target_size is None:
            target_size = TARGET_SIZE
            
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"[ERROR] Image preprocessing failed: {str(e)}")
        raise

def generate_gradcam_heatmap(model, img_array, pred_index=None):
    """Generate Grad-CAM heatmap - Robust version"""
    try:
        # Find the last convolutional layer
        last_conv_layer = None
        last_conv_layer_name = None
        
        for layer in reversed(model.layers):
            # Check if it's a Conv2D layer
            if 'conv' in layer.name.lower():
                last_conv_layer = layer
                last_conv_layer_name = layer.name
                break
        
        if last_conv_layer is None:
            print("[WARNING] No convolutional layer found for Grad-CAM")
            return None
        
        print(f"[INFO] Using convolutional layer: {last_conv_layer_name}")
        
        # Build a model that returns both conv layer output and final prediction
        grad_model = keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        # Compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = 0
            loss = predictions[:, pred_index]
        
        # Extract filters and gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Check if gradients are None
        if grads is None:
            print("[ERROR] Gradients are None - cannot compute Grad-CAM")
            return None
        
        # Compute the mean intensity of the gradient over each feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel in the feature map array
        # by "how important this channel is"
        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()
        
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # The channel-wise mean of the resulting feature map
        # is our heatmap of class activation
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # Normalize the heatmap
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        
        if max_val > 0:
            heatmap = heatmap / max_val
        else:
            print("[WARNING] Heatmap max value is 0")
            return None
        
        print(f"[OK] Grad-CAM heatmap generated successfully. Shape: {heatmap.shape}")
        return heatmap
        
    except Exception as e:
        print(f"[ERROR] Grad-CAM generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_heatmap_overlay(image_path, heatmap):
    """Create overlay of heatmap on original image"""
    try:
        # Read original image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
        
        return overlay
    except Exception as e:
        print(f"[ERROR] Overlay creation failed: {str(e)}")
        return None

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'input_size': TARGET_SIZE
    })

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Analyze single image for deepfake detection"""
    try:
        print("\n" + "="*60)
        print("[INFO] Starting image analysis...")
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename, 'image'):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"[INFO] Image saved: {filepath}")
        print(f"[INFO] File exists: {os.path.exists(filepath)}")
        print(f"[INFO] File size: {os.path.getsize(filepath)} bytes")
        
        # Preprocess and predict
        img_array = preprocess_image(filepath)
        prediction = model.predict(img_array, verbose=0)
        confidence = float(prediction[0][0])
        
        # Determine result
        is_real = confidence > 0.5
        label = 'Real' if is_real else 'Fake'
        percentage = confidence * 100 if is_real else (1 - confidence) * 100
        
        print(f"[RESULT] Image: {label} ({percentage:.2f}%)")
        
        # Generate Grad-CAM
        print("[INFO] Generating Grad-CAM heatmap...")
        heatmap = generate_gradcam_heatmap(model, img_array)
        
        # Prepare result
        result = {
            'prediction': label,
            'confidence': float(confidence),
            'percentage': float(percentage),
            'real_probability': float(confidence),
            'fake_probability': float(1 - confidence),
            'image_path': f'/static/uploads/{filename}'
        }
        
        # Save XAI visualization if heatmap was generated
        if heatmap is not None:
            print("[INFO] Creating heatmap overlay...")
            overlay = create_heatmap_overlay(filepath, heatmap)
            if overlay is not None:
                xai_filename = f"xai_{filename}"
                xai_path = os.path.join(app.config['OUTPUT_FOLDER'], xai_filename)
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(xai_path, overlay_bgr)
                if success:
                    result['xai_image'] = f'/static/outputs/{xai_filename}'
                    print(f"[OK] XAI image saved: {xai_path}")
                else:
                    print(f"[ERROR] Failed to save XAI image to: {xai_path}")
            else:
                print("[ERROR] Could not create heatmap overlay")
        else:
            print("[WARNING] Could not generate Grad-CAM heatmap")
        
        # Convert all numpy types to Python types
        result = convert_to_python_type(result)
        
        return jsonify(result), 200
    
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_video_xai', methods=['POST'])
def analyze_video_xai():
    """Analyze video with XAI visualization"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename, 'video'):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"[INFO] Video saved: {filepath}")
        
        # Create output video path
        output_filename = f"analyzed_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        print(f"[INFO] Writing output to: {output_path}")
        
        # Open video
        cap = cv2.VideoCapture(filepath)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"[INFO] Processing {total_frames} frames...")
        
        frame_predictions = []
        frame_count = 0
        process_every_n_frames = max(1, fps // 3)  # Process 3 frames per second
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame for prediction
            if frame_count % process_every_n_frames == 0:
                # Save frame temporarily
                temp_frame_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_frame.jpg')
                cv2.imwrite(temp_frame_path, frame)
                
                # Predict
                img_array = preprocess_image(temp_frame_path)
                prediction = model.predict(img_array, verbose=0)
                confidence = float(prediction[0][0])
                frame_predictions.append(confidence)
                
                # Generate heatmap
                if frame_count % (process_every_n_frames * 10) == 0:  # Log every 10th processed frame
                    heatmap = generate_gradcam_heatmap(model, img_array)
                    if heatmap is None:
                        print(f"[WARNING] Heatmap failed for frame {frame_count}: Heatmap generation failed")
                
                # Clean up temp file
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)
                
                # Add text overlay
                label = 'Real' if confidence > 0.5 else 'Fake'
                color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)
                cv2.putText(frame, f'{label}: {confidence*100:.1f}%', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Write frame
            out.write(frame)
            frame_count += 1
            
            # Progress update
            if frame_count % 50 == 0:
                print(f"[INFO] Progress: {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        print(f"[OK] Output video saved: {output_path}")
        
        # Calculate statistics
        if frame_predictions:
            avg_confidence = float(np.mean(frame_predictions))
            std_confidence = float(np.std(frame_predictions))
        else:
            avg_confidence = 0.5
            std_confidence = 0.0
        
        is_real = avg_confidence > 0.5
        label = 'Real' if is_real else 'Fake'
        percentage = avg_confidence * 100 if is_real else (1 - avg_confidence) * 100
        
        print(f"[RESULT] Video: {label} (avg score: {avg_confidence:.4f})")
        print(f"[RESULT] Video XAI: {label} ({percentage:.2f}%)")
        
        result = {
            'prediction': label,
            'confidence': float(avg_confidence),
            'percentage': float(percentage),
            'std_deviation': float(std_confidence),
            'total_frames': total_frames,
            'processed_frames': len(frame_predictions),
            'output_video': f'/static/outputs/{output_filename}',
            'frame_predictions': [float(x) for x in frame_predictions]
        }
        
        # Convert to Python types
        result = convert_to_python_type(result)
        
        return jsonify(result), 200
    
    except Exception as e:
        print(f"[ERROR] Video analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:folder>/<path:filename>')
def serve_static(folder, filename):
    """Serve static files"""
    folder_path = os.path.join('static', folder)
    file_path = os.path.join(folder_path, filename)
    
    if os.path.exists(file_path):
        return send_file(file_path)
    else:
        print(f"[ERROR] File not found: {file_path}")
        return jsonify({'error': 'File not found'}), 404

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    """Analyze video without XAI"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename, 'video'):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"[INFO] Video saved: {filepath}")
        
        # Open video
        cap = cv2.VideoCapture(filepath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        frame_predictions = []
        frame_count = 0
        process_every_n_frames = max(1, fps // 3)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % process_every_n_frames == 0:
                temp_frame_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_frame.jpg')
                cv2.imwrite(temp_frame_path, frame)
                
                img_array = preprocess_image(temp_frame_path)
                prediction = model.predict(img_array, verbose=0)
                confidence = float(prediction[0][0])
                frame_predictions.append(confidence)
                
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)
            
            frame_count += 1
        
        cap.release()
        
        # Calculate statistics
        avg_confidence = float(np.mean(frame_predictions))
        is_real = avg_confidence > 0.5
        label = 'Real' if is_real else 'Fake'
        percentage = avg_confidence * 100 if is_real else (1 - avg_confidence) * 100
        
        result = {
            'prediction': label,
            'confidence': float(avg_confidence),
            'percentage': float(percentage),
            'total_frames': total_frames,
            'processed_frames': len(frame_predictions)
        }
        
        result = convert_to_python_type(result)
        
        return jsonify(result), 200
    
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/extract_keyframes', methods=['POST'])
def extract_keyframes():
    """Extract keyframes from video"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        num_frames = int(request.form.get('num_frames', 10))
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        cap = cv2.VideoCapture(filepath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Select frame indices
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        keyframes = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                keyframe_filename = f'keyframe_{idx}_{filename}.jpg'
                keyframe_path = os.path.join(app.config['KEYFRAMES_FOLDER'], keyframe_filename)
                cv2.imwrite(keyframe_path, frame)
                keyframes.append(f'/static/keyframes/{keyframe_filename}')
        
        cap.release()
        
        result = {
            'keyframes': keyframes,
            'count': len(keyframes)
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Deepfake Detection with XAI - Server Starting")
    print("=" * 60)
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Keyframes folder: {KEYFRAMES_FOLDER}")
    print(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB")
    print(f" Image formats: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}")
    print(f"Video formats: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}")
    print("=" * 60)
    print()
    
    # Load model
    if load_model():
        print("[OK] Model loaded successfully")
        print("\n" + "=" * 60)
        print(f"Server running on http://localhost:5000")
        print(f"Using input size: {TARGET_SIZE}")
        print("=" * 60 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    else:
        print("[ERROR] Failed to load model. Server not started.")
        sys.exit(1)
