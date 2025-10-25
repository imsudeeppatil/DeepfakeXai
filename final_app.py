import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from utils.model_utils import get_model
from utils.xai_utils import make_gradcam_heatmap, overlay_heatmap_on_image

# --- Best Practice: Load the model once at startup ---
# This prevents reloading the model from disk on every single request.
print("Loading model... Please wait.")
model = get_model()
# Build the model once here, so it's ready for all subsequent requests.
model.build((None, 224, 224, 3)) 
print("Model loaded and built successfully.")
# ----------------------------------------------------

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
GRADCAM_FOLDER = os.path.join(BASE_DIR, "static", "gradcam")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    # The model is already loaded, so no need for the try/except block here.
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "media" not in request.files:
        return "No file part", 400
    file = request.files["media"]
    if file.filename == "":
        return "No selected file", 400

    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return "Invalid file format. Please upload JPG or PNG.", 400

    save_path = os.path.join(app.config.get("UPLOAD_FOLDER", UPLOAD_FOLDER), filename)
    file.save(save_path)

    original_bgr = cv2.imread(save_path)
    if original_bgr is None:
        return "Failed to read uploaded image.", 400

    # The model is already loaded globally, no need to call get_model() again.
    
    # Preprocess the image for the model
    img_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224)).astype("float32") / 255.0
    input_arr = np.expand_dims(img_resized, axis=0) # Shape: (1, 224, 224, 3)

    # Get prediction
    pred_score = float(model.predict(input_arr, verbose=0)[0][0])
    label = "Fake" if pred_score > 0.5 else "Real"

    # Grad-CAM: This will now work correctly because the model is built.
    heatmap = make_gradcam_heatmap(model, input_arr)
    overlay = overlay_heatmap_on_image(original_bgr, heatmap, alpha=0.45)
    
    gradcam_filename = filename.rsplit(".", 1)[0] + "_gradcam.jpg"
    gradcam_path = os.path.join(GRADCAM_FOLDER, gradcam_filename)
    cv2.imwrite(gradcam_path, overlay)

    # Prepare paths for the template
    media_rel = os.path.join("uploads", filename).replace("\\", "/")
    heatmap_rel = os.path.join("gradcam", gradcam_filename).replace("\\", "/")

    return render_template(
        "result.html",
        media_path=media_rel,
        heatmap_path=heatmap_rel,
        result=label,
        score=round(pred_score, 3),
    )

if __name__ == "__main__":
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.run(debug=True)