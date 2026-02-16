import os
import io
import json
import base64
import numpy as np
import cv2
from PIL import Image
from collections import OrderedDict

import tensorflow as tf
from keras.utils import load_img, img_to_array

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------
# Flask App
# ----------------------------
app = Flask(__name__, template_folder="../templates")
CORS(app)

# ----------------------------
# Globals / Lazy-loaded models
# ----------------------------
glaucoma_model = None
epilepsy_model = None
model_loaded_flag = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------
# Glaucoma Model (Keras)
# ----------------------------
def load_glaucoma_model():
    global glaucoma_model
    if glaucoma_model is None:
        model_path = os.path.join(BASE_DIR, "glaucoma_model.keras")
        glaucoma_model = tf.keras.models.load_model(model_path)
    return glaucoma_model

TARGET_SIZE = (256, 256)

# ----------------------------
# Epilepsy Model (PyTorch)
# ----------------------------
class EpilepsyDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(EpilepsyDetectionModel, self).__init__()
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.efficientnet(x)

def load_epilepsy_model():
    global epilepsy_model
    if epilepsy_model is None:
        model_path = os.path.join(BASE_DIR, "epilepsy_detection_model.pth")
        epilepsy_model = EpilepsyDetectionModel(num_classes=2)
        try:
            state_dict = torch.load(model_path, map_location="cpu")
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if not k.startswith("efficientnet."):
                    k = "efficientnet." + k
                new_state_dict[k] = v
            epilepsy_model.load_state_dict(new_state_dict, strict=False)
            epilepsy_model.eval()
            print("✅ Epilepsy model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading epilepsy model: {e}")
            epilepsy_model = None
    return epilepsy_model

INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
CLASS_NAMES = ['healthy', 'epilepsy']

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("dash.html")

@app.route("/brain_tumor.html")
def brain_tumor_page():
    return render_template("brain_tumor.html")

@app.route("/heart_prediction.html")
def heart_prediction_page():
    return render_template("heart_prediction.html")

@app.route("/eye_glaucoma.html")
def eye_glaucoma_page():
    return render_template("eye_glaucoma.html")

# ----------------------------
# Glaucoma Prediction
# ----------------------------
@app.route('/predict', methods=['POST'])
def predict_glaucoma():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = load_img(io.BytesIO(file.read()), target_size=TARGET_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        model = load_glaucoma_model()
        prediction = model.predict(img_array)[0][0]
        result = 'Glaucoma' if prediction > 0.5 else 'No Glaucoma'

        return jsonify({
            'prediction': result,
            'probability': float(prediction)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ----------------------------
# Epilepsy Prediction
# ----------------------------
def generate_cam_heatmap(model, img_tensor, target_class_idx):
    model.eval()
    activations, gradients = None, None

    def save_activations(module, input, output):
        nonlocal activations
        activations = output

    def save_gradients(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    target_layer = model.efficientnet.features[-1]
    hook_fwd = target_layer.register_forward_hook(save_activations)
    hook_bwd = target_layer.register_full_backward_hook(save_gradients)

    img_input = img_tensor.unsqueeze(0)
    output = model(img_input)
    probabilities = torch.softmax(output, dim=1)
    predicted_class_idx = torch.argmax(probabilities, dim=1).item()
    model_confidence = probabilities[0, predicted_class_idx].item()

    if CLASS_NAMES[predicted_class_idx] == 'healthy':
        hook_fwd.remove()
        hook_bwd.remove()
        return None, predicted_class_idx, model_confidence

    model.zero_grad()
    target_output = output[0, predicted_class_idx]
    target_output.backward()

    gradients_np = gradients.cpu().data.numpy()[0]
    activations_np = activations.cpu().data.numpy()[0]
    weights = np.mean(gradients_np, axis=(1, 2))
    cam = np.zeros(activations_np.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activations_np[i]
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam) if np.max(cam) > 0 else np.zeros(cam.shape)
    hook_fwd.remove()
    hook_bwd.remove()
    return cam, predicted_class_idx, model_confidence

def overlay_heatmap_on_image(original_img, heatmap):
    original_img_np = np.array(original_img)
    heatmap_resized = cv2.resize(heatmap, (original_img_np.shape[1], original_img_np.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(original_img_np, 0.6, heatmap_colored, 0.4, 0)
    return Image.fromarray(superimposed)

@app.route('/brain', methods=['POST'])
def predict_epilepsy():
    model = load_epilepsy_model()
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
        img_tensor = INFERENCE_TRANSFORMS(img)
        original_img = img.copy()

        with torch.no_grad():
            outputs = model(img_tensor.unsqueeze(0))
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(outputs, dim=1).item()
            model_confidence = probabilities[0, predicted_class_idx].item()
            predicted_class_name = CLASS_NAMES[predicted_class_idx]

        heatmap_base64 = None
        severity_score, severity_category = 0.0, "N/A"

        if predicted_class_name == 'epilepsy':
            heatmap, _, _ = generate_cam_heatmap(model, img_tensor, predicted_class_idx)
            if heatmap is not None:
                blended_img = overlay_heatmap_on_image(original_img, heatmap)
                buffered = io.BytesIO()
                blended_img.save(buffered, format="PNG")
                heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'prediction': predicted_class_name,
            'confidence': f"{model_confidence:.4f}",
            'severity_score': f"{severity_score:.2f}",
            'severity_category': severity_category,
            'heatmap': heatmap_base64
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({'error': 'Error processing image'}), 500

# ----------------------------
# Run Flask App
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
