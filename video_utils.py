import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input

model = load_model("C:\Major Project Backup\cnn_fake_detector.h5")

def predict_image(img):
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    return pred  # return raw score for aggregation

def predict_video(video_path, frame_interval=10, threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    frame_preds = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            pred = predict_image(frame)
            frame_preds.append(pred)
        frame_idx += 1

    cap.release()
    if len(frame_preds) == 0:
        return "Unable to process video."

    avg_score = np.mean(frame_preds)
    result = "Fake" if avg_score > threshold else "Real"
    return result, avg_score
