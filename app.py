import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib
matplotlib.use('Agg')  # Set the non-interactive backend
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from scipy.ndimage import binary_dilation, binary_erosion

app = Flask(__name__)
app.template_folder = "templates"

# Define paths
MODEL_PATH = "models/my_segmentation_model_100_efficientnet.h5"
UPLOAD_FOLDER = "uploads"
HEATMAP_FOLDER = "static/heatmaps"

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route("/")
def home():
    return render_template("dashboard.html")

# Image preprocessing function
def preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    if image is None:
        raise ValueError("Error loading image. Check the file format.")

    image = cv2.resize(image, target_size)  # Resize first
    image = np.expand_dims(image, axis=-1)  # Expand channel: (128, 128, 1)
    image = np.repeat(image, 3, axis=-1)  # Convert to (128, 128, 3)
    
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Postprocess predicted mask
def postprocess_mask(mask, threshold=0.01):

    mask = (mask > threshold).astype(np.uint8)  # Convert to binary (0 or 1)
    return mask

# Classify tumor presence
def classify_tumor(y_pred, threshold=0.3, min_tumor_size=10):
    binary_pred_mask = (y_pred > threshold).astype(np.uint8)
    predicted_tumor_pixels = np.sum(binary_pred_mask)
    return "Yes" if predicted_tumor_pixels > min_tumor_size else "No"

def compute_tumor_size(mask):
    return np.sum(mask)  # Count tumor pixels in the mask

def classify_tumor_severity(tumor_percentage, mild_threshold, moderate_threshold):
    if tumor_percentage <= mild_threshold:
        return "Mild"
    elif tumor_percentage <= moderate_threshold:
        return "Moderate"
    else:
        return "Severe"

def generate_heatmap(original_image, mask, filename, image_size=(128, 128)):
    HEATMAP_FOLDER = "static/heatmaps"
    os.makedirs(HEATMAP_FOLDER, exist_ok=True)
    
    # Ensure mask is properly thresholded
    mask = (mask > 0.01).astype(np.uint8)  # Adjust threshold if needed
    tumor_size = np.sum(mask)
    tumor_percentage = (tumor_size / (image_size[0] * image_size[1])) * 100
    
    # Compute severity thresholds dynamically
    mild_threshold = 5  # Example thresholds (adjust as needed)
    moderate_threshold = 15
    
    def classify_tumor_severity(tumor_percentage, mild_threshold, moderate_threshold):
        if tumor_percentage <= mild_threshold:
            return "Mild"
        elif tumor_percentage <= moderate_threshold:
            return "Moderate"
        else:
            return "Severe"
    
    severity_label = classify_tumor_severity(tumor_percentage, mild_threshold, moderate_threshold)
    
    colormap_dict = {
        "Mild": cv2.COLORMAP_COOL,
        "Moderate": cv2.COLORMAP_JET,
        "Severe": cv2.COLORMAP_HOT
    }
    selected_colormap = colormap_dict[severity_label]
    
    # Convert mask to heatmap
    heatmap_resized = cv2.resize((mask * 255).astype(np.uint8), (image_size[1], image_size[0]))
    heatmap = cv2.applyColorMap(heatmap_resized, selected_colormap)
    
    # Ensure original image is resized to match heatmap
    if original_image.shape[:2] != (image_size[0], image_size[1]):
        original_image = cv2.resize(original_image, (image_size[1], image_size[0]))
    
    # Convert grayscale image to 3-channel if needed
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    
    # Mask the heatmap to only apply colors where the tumor is detected
    masked_heatmap = cv2.bitwise_and(heatmap, heatmap, mask=(mask * 255).astype(np.uint8))
    
    # Overlay only tumor regions
    overlay = cv2.addWeighted(original_image, 0.6, masked_heatmap, 0.4, 0)
    
    heatmap_path = os.path.join(HEATMAP_FOLDER, f"heatmap_{filename}.png")
    cv2.imwrite(heatmap_path, overlay)
    
    # Debugging: Save intermediate mask output
    cv2.imwrite(os.path.join(HEATMAP_FOLDER, f"mask_{filename}.png"), (mask * 255).astype(np.uint8))
    
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original_image.squeeze(), cmap="gray")
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze(), cmap="jet", alpha=0.6)
    plt.title("Tumor Mask Heatmap")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Overlay Heatmap\nSeverity: {severity_label} ({tumor_percentage:.2f}%)")
    plt.axis("off")
    
    plt.show()
    
    return heatmap_path  # Return saved path

# Visualization function (for debugging)
def visualize_prediction(image, mask):
    plt.figure(figsize=(12, 4))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    # Predicted Mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze(), cmap="jet", alpha=0.6)  # Use 'jet' for heatmap
    plt.title("Predicted Mask (Heatmap)")
    plt.axis("off")

    # Overlay Mask on Image
    plt.subplot(1, 3, 3)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.imshow(mask.squeeze(), cmap="jet", alpha=0.5)  # Overlay heatmap
    plt.title("Overlay on Image")
    plt.axis("off")

    plt.show()

# Define loss functions
def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    smooth = tf.keras.backend.epsilon()
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, smooth, 1.0 - smooth)
    tp = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=[1, 2, 3])
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=[1, 2, 3])
    tversky_index = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return tf.reduce_mean(1 - tversky_index)

def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=1.3):
    tversky = tversky_loss(y_true, y_pred, alpha, beta)
    focal_tversky = tf.math.log(tf.math.cosh(tversky * gamma))
    return focal_tversky

def combined_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=1.3, weight=0.5):
    dice = 1 - (2 * tf.reduce_sum(y_true * y_pred) + 1e-6) / \
           (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-6)
    tversky = focal_tversky_loss(y_true, y_pred, alpha, beta, gamma)
    return weight * dice + (1 - weight) * tversky

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    image = preprocess_image(filepath)
    if image is None:
        return jsonify({"error": "Image preprocessing failed"}), 500

    prediction = model.predict(image)[0]  # Get predicted mask

    # Load original image for heatmap
    original_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        return jsonify({"error": "Error loading original image"}), 500

    # Generate heatmap and get path
    heatmap_path = generate_heatmap(original_image, prediction, filename)

    result = classify_tumor(prediction)

    return jsonify({
        "tumor": result,
        "heatmap_url": f"/{heatmap_path}"  # Return heatmap URL
    })

if __name__ == "__main__":
    app.run(debug=True)
