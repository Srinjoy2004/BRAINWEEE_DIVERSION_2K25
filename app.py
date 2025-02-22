import numpy as np
import tensorflow as tf
import cv2
import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Set the folder for templates
app.template_folder = "templates"

# Print TensorFlow & Keras versions for debugging
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")

# Load trained VGG19 U-Net model with error handling
MODEL_PATH = "models/segmentation_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Set model to None if loading fails

# Configure upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Serve the default dashboard.html when opening the root URL
@app.route("/")
def home():
    return render_template("dashboard.html")

# Preprocessing function
def preprocess_image(image_path, target_size=(128, 128)):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Error loading image. Check the file format.")

        image = cv2.resize(image, target_size)
        image = image / 255.0  # Normalize
        image = np.expand_dims(image, axis=(0, -1))  # Reshape for model [1, H, W, 1]
        return image
    except Exception as e:
        print(f"❌ Preprocessing Error: {e}")
        return None

# Tumor classification function
def classify_tumor(y_pred, threshold=0.1, min_tumor_size=20):
    binary_pred_mask = (y_pred > threshold).astype(np.uint8)
    predicted_tumor_pixels = np.sum(binary_pred_mask)
    return "Yes" if predicted_tumor_pixels > min_tumor_size else "No"

# API Route to predict tumor
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Process the image
    image = preprocess_image(filepath)
    if image is None:
        return jsonify({"error": "Image preprocessing failed"}), 500

    # Run the model if it is loaded
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    prediction = model.predict(image)
    result = classify_tumor(prediction)

    return jsonify({"tumor": result})

if __name__ == "__main__":
    app.run(debug=True)
