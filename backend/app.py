import json
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
import numpy as np 

import tensorflow as tf
from keras.utils import load_img, img_to_array

import io

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

from collections import OrderedDict
import cv2
import base64

app = Flask(__name__,template_folder="../templates")
CORS(app)
# --- Helper Function to fix Plotly transparency issue ---
def hex_to_rgba(hex_color, alpha_percent):
    """Converts a 6-digit hex color to an rgba string with transparency."""
    hex_color = hex_color.lstrip('#')
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        alpha = alpha_percent / 100.0
        return f'rgba({r},{g},{b},{alpha})'
    except ValueError:
        return 'rgba(128, 128, 128, 0.2)' 
# --------------------------------------------------------

class EnhancedCardioRiskAssessment:
    def __init__(self):
        self.risk_levels = {
            'Low Risk': {'min': 0, 'max': 5, 'color': '#2ecc71', 'emoji': '🟢', 'name': 'Low Risk'},
            'Moderate Risk': {'min': 6, 'max': 10, 'color': '#f39c12', 'emoji': '🟡', 'name': 'Moderate Risk'},
            'High Risk': {'min': 11, 'max': 15, 'color': '#e74c3c', 'emoji': '🟠', 'name': 'High Risk'},
            'Very High Risk': {'min': 16, 'max': 100, 'color': '#c0392b', 'emoji': '🔴', 'name': 'Very High Risk'}
        }
        
    def _calculate_component_scores(self, patient_data):
        # ... (Omitted for brevity - Scoring logic remains the same) ...
        age = patient_data['age']
        sys_bp = patient_data['systolic_bp']
        dia_bp = patient_data['diastolic_bp']
        tot_chol = patient_data['total_cholesterol']
        ldl = patient_data['ldl_cholesterol']
        hdl = patient_data['hdl_cholesterol']
        troponin = patient_data['troponin']
        triglycerides = patient_data['triglycerides']
        heart_rate = patient_data['heart_rate']
        sex = patient_data['sex']

        score_breakdown = {}
        total_score = 0

        # 1. Age scoring
        age_score = 0
        if age >= 40: age_score += 2
        if age >= 50: age_score += 3
        if age >= 60: age_score += 2
        score_breakdown['Age'] = age_score
        total_score += age_score

        # 2. Sex scoring
        sex_score = 1 if sex == 'Male' else 0
        score_breakdown['Sex'] = sex_score
        total_score += sex_score

        # 3. Blood pressure scoring
        bp_score = 0
        if sys_bp >= 120: bp_score += 1
        if sys_bp >= 140: bp_score += 3
        if sys_bp >= 160: bp_score += 2
        if sys_bp >= 180: bp_score += 5
        if dia_bp <= 60: bp_score += 2
        if dia_bp >= 90: bp_score += 1
        if dia_bp >= 100: bp_score += 2
        score_breakdown['Blood Pressure'] = bp_score
        total_score += bp_score

        # 4. Cholesterol scoring
        cholesterol_score = 0
        if tot_chol >= 200: cholesterol_score += 2
        if tot_chol >= 240: cholesterol_score += 1
        if ldl >= 100: cholesterol_score += 1
        if ldl >= 130: cholesterol_score += 1
        if ldl >= 160: cholesterol_score += 3
        if hdl < 40: cholesterol_score += 2
        if hdl >= 60: cholesterol_score -= 1  # Protective
        score_breakdown['Cholesterol'] = cholesterol_score
        total_score += cholesterol_score

        # 5. Troponin scoring (cardiac marker)
        troponin_score = 0
        if troponin > 0.04: troponin_score += 5
        if troponin > 0.1: troponin_score += 2
        if troponin > 0.5: troponin_score += 8
        if troponin > 1.0: troponin_score += 5
        score_breakdown['Troponin'] = troponin_score
        total_score += troponin_score

        # 6. Additional factors
        additional_score = 0
        if triglycerides >= 150: additional_score += 1
        if triglycerides >= 200: additional_score += 1
        if heart_rate >= 100: additional_score += 1
        if heart_rate < 60: additional_score += 1
        score_breakdown['Other Factors'] = additional_score
        total_score += additional_score

        final_score = min(total_score, 45)
        return final_score, score_breakdown
    
    def calculate_comprehensive_risk_score(self, patient_data):
        return self._calculate_component_scores(patient_data)

    def get_risk_level(self, score):
        for level, info in self.risk_levels.items():
            if info['min'] <= score <= info['max']:
                return info
        return self.risk_levels['Very High Risk']

    def get_detailed_recommendations(self, risk_level_info, patient_data, score_breakdown):
        # ... (Omitted for brevity - Recommendation logic remains the same) ...
        risk_level = risk_level_info['name']
        
        base_recommendations = {
            'Low Risk': {'title': '🟢 Continue Preventive Care', 'actions': ['Maintain healthy lifestyle...', 'Regular physical activity...', 'Annual check-ups and blood pressure monitoring', 'Maintain healthy weight and avoid smoking']},
            'Moderate Risk': {'title': '🟡 Lifestyle Modification & Monitoring', 'actions': ['Consult cardiologist within 2-4 weeks...', 'Implement dietary changes (reduce salt, saturated fats)', 'Increase physical activity gradually', 'Monitor blood pressure twice weekly', 'Consider stress management techniques']},
            'High Risk': {'title': '🟠 Immediate Medical Attention Required', 'actions': ['Urgent cardiology consultation within 1 week...', 'Begin medication therapy under supervision', 'Frequent monitoring (BP 2x daily, symptoms)', 'Cardiac testing (ECG, Echo, Stress test)', 'Strict dietary and activity modifications']},
            'Very High Risk': {'title': '🔴 EMERGENCY - Immediate Intervention Needed', 'actions': ['Seek emergency medical care immediately...', 'High probability of acute cardiac event', 'Do not delay - call emergency services if symptoms worsen', 'Continuous cardiac monitoring required', 'Prepare for possible hospitalization']}
        }
        
        base_rec = base_recommendations.get(risk_level, base_recommendations['Low Risk'])

        critical_alerts = []
        troponin = patient_data['troponin']
        sys_bp = patient_data['systolic_bp']
        dia_bp = patient_data['diastolic_bp']
        ldl = patient_data['ldl_cholesterol']
        age = patient_data['age']

        if sys_bp >= 180 or dia_bp >= 120:
            critical_alerts.append("🚨 **HYPERTENSIVE CRISIS:** Blood pressure critically high. Seek immediate care.")
        if troponin > 0.04:
            critical_alerts.append("🚨 **CARDIAC ENZYMES:** Elevated Troponin indicates potential myocardial damage.")
        if dia_bp <= 50:
            critical_alerts.append("🚨 **HYPOTENSION:** Dangerously low diastolic pressure.")
        if ldl >= 160:
             critical_alerts.append("🚨 **VERY HIGH LDL:** Aggressive lipid management needed.")

        factor_recommendations = []
        if score_breakdown.get('Blood Pressure', 0) >= 5:
            factor_recommendations.append("• Focus on strict BP control with medication and lifestyle.")
        if score_breakdown.get('Troponin', 0) > 0:
            factor_recommendations.append("• Cardiac enzymes elevated - immediate workup for Acute Coronary Syndrome (ACS) required.")
        if score_breakdown.get('Cholesterol', 0) >= 3:
            factor_recommendations.append("• Lipid management: Consult doctor regarding statin therapy due to high risk profile.")
        if age >= 50:
            factor_recommendations.append("• Age >50: Regular preventative cardiac screening is essential.")
        
        return base_rec, critical_alerts, factor_recommendations


    def create_comprehensive_visualization(self, score, risk_level_info, patient_data, score_breakdown):
        
        # 1. Setup Layout
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "indicator"}, {"type": "heatmap"}],
                   [{"type": "scatterpolar"}, {"type": "bar"}]],
            subplot_titles=('Risk Level Gauge', 'Risk Heat Map',
                            'Parameter Analysis', 'Risk Factor Breakdown'),
            # Crucial adjustment for tighter spacing
            vertical_spacing=0.08, 
            horizontal_spacing=0.1
        )
        
        # Data needed for plotting (already cast to numeric types)
        sys_bp = patient_data['systolic_bp']
        dia_bp = patient_data['diastolic_bp']
        tot_chol = patient_data['total_cholesterol']
        hdl = patient_data['hdl_cholesterol']
        troponin = patient_data['troponin']
        age = patient_data['age']
        
        # Determine colors for gauge steps
        low_color = self.risk_levels['Low Risk']['color']
        mod_color = self.risk_levels['Moderate Risk']['color']
        high_color = self.risk_levels['High Risk']['color']
        vhigh_color = self.risk_levels['Very High Risk']['color']


        # 1. Risk gauge (Row 1, Col 1)
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Score: {risk_level_info['emoji']} {risk_level_info['name']}"},
            gauge = {
                'axis': {'range': [None, 35]},
                'bar': {'color': risk_level_info['color']},
                'steps': [
                    {'range': [0, 5], 'color': hex_to_rgba(low_color, 20)},
                    {'range': [5, 10], 'color': hex_to_rgba(mod_color, 20)},
                    {'range': [10, 15], 'color': hex_to_rgba(high_color, 20)},
                    {'range': [15, 35], 'color': hex_to_rgba(vhigh_color, 20)}
                ],
                'threshold': {
                    'line': {'color': risk_level_info['color'], 'width': 4},
                    'thickness': 0.75,
                    'value': score}
            }
        ), row=1, col=1)

        # 2. Risk factor breakdown (Row 2, Col 2) - Horizontal Bar Chart
        factors = list(score_breakdown.keys())
        scores = list(score_breakdown.values())
        colors = ['#3498db' if s <= 2 else '#e74c3c' if s <= 5 else '#c0392b' for s in scores]

        fig.add_trace(go.Bar(
            x=scores, 
            y=factors, 
            orientation='h', 
            marker_color=colors,
            text=scores,
            textposition='auto',
        ), row=2, col=2)
        
        # 3. Radar chart for parameters (Row 2, Col 1)
        categories = ['BP Status', 'Lipid Profile', 'Cardiac Markers', 'Age Risk', 'Overall Score']

        # Normalized values (0-10 scale)
        bp_status = min(10, (max(0, sys_bp - 110) / 7 + max(0, 80 - dia_bp) / 3))
        lipid_profile = min(10, (max(0, tot_chol - 150) / 10 + max(0, 160 - hdl) / 4))
        cardiac_markers = min(10, troponin * 80) if troponin < 1 else 10 
        age_risk = min(10, age / 8)
        overall_score = min(10, score / 3) 

        values = [bp_status, lipid_profile, cardiac_markers, age_risk, overall_score]
        values += values[:1] 

        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name='Patient Status',
            line=dict(color=risk_level_info['color'])
        ), row=2, col=1)

        # 4. Heatmap for risk comparison (Row 1, Col 2)
        risk_parameters = ['Sys BP', 'Dia BP', 'Tot Chol', 'LDL', 'HDL', 'Troponin']
        risk_values = [
            sys_bp / 180,
            (100 - dia_bp) / 70 if dia_bp < 70 else dia_bp / 100,
            tot_chol / 200,
            patient_data['ldl_cholesterol'] / 130,
            (60 - hdl) / 40,
            troponin * 20
        ]
        
        risk_values = [min(max(v, 0), 2) for v in risk_values]

        fig.add_trace(go.Heatmap(
            z=[risk_values],
            x=risk_parameters,
            y=['Risk'],
            colorscale='RdYlGn_r', 
            showscale=True,
            hoverongaps=False
        ), row=1, col=2)

        # 5. Final Layout Adjustments (Fixing overlap/padding)
        fig.update_layout(
            # Crucially reduced height to minimize blank space
            height=650, 
            margin=dict(l=50, r=50, t=50, b=50), 
            title_text=f"Comprehensive Cardiovascular Risk Analysis: {risk_level_info['name']}",
            showlegend=False,
            template="plotly_dark", 
            font={'color': '#f8fafc', 'family': 'Inter'}
        )
        
        # Explicitly setting annotation/title positions to prevent shifting/overlap
        # These coordinates are optimized for the new height=650 and tighter spacing
        fig.layout.annotations[0].update(y=1.05)
        fig.layout.annotations[1].update(y=1.05)
        fig.layout.annotations[2].update(y=0.47)
        fig.layout.annotations[3].update(y=0.47)


        return fig

    # --- (Remaining helper methods and Flask route omitted for brevity, they are correct) ---
    def calculate_comprehensive_risk_score(self, patient_data):
        return self._calculate_component_scores(patient_data)

    def get_detailed_recommendations(self, risk_level_info, patient_data, score_breakdown):
        # ... (Recommendation logic omitted for brevity) ...
        risk_level = risk_level_info['name']
        
        base_recommendations = {
            'Low Risk': {'title': '🟢 Continue Preventive Care', 'actions': ['Maintain healthy lifestyle...', 'Regular physical activity...', 'Annual check-ups and blood pressure monitoring', 'Maintain healthy weight and avoid smoking']},
            'Moderate Risk': {'title': '🟡 Lifestyle Modification & Monitoring', 'actions': ['Consult cardiologist within 2-4 weeks...', 'Implement dietary changes (reduce salt, saturated fats)', 'Increase physical activity gradually', 'Monitor blood pressure twice weekly', 'Consider stress management techniques']},
            'High Risk': {'title': '🟠 Immediate Medical Attention Required', 'actions': ['Urgent cardiology consultation within 1 week...', 'Begin medication therapy under supervision', 'Frequent monitoring (BP 2x daily, symptoms)', 'Cardiac testing (ECG, Echo, Stress test)', 'Strict dietary and activity modifications']},
            'Very High Risk': {'title': '🔴 EMERGENCY - Immediate Intervention Needed', 'actions': ['Seek emergency medical care immediately...', 'High probability of acute cardiac event', 'Do not delay - call emergency services if symptoms worsen', 'Continuous cardiac monitoring required', 'Prepare for possible hospitalization']}
        }
        
        base_rec = base_recommendations.get(risk_level, base_recommendations['Low Risk'])

        critical_alerts = []
        troponin = patient_data['troponin']
        sys_bp = patient_data['systolic_bp']
        dia_bp = patient_data['diastolic_bp']
        ldl = patient_data['ldl_cholesterol']
        age = patient_data['age']

        if sys_bp >= 180 or dia_bp >= 120:
            critical_alerts.append("🚨 **HYPERTENSIVE CRISIS:** Blood pressure critically high. Seek immediate care.")
        if troponin > 0.04:
            critical_alerts.append("🚨 **CARDIAC ENZYMES:** Elevated Troponin indicates potential myocardial damage.")
        if dia_bp <= 50:
            critical_alerts.append("🚨 **HYPOTENSION:** Dangerously low diastolic pressure.")
        if ldl >= 160:
             critical_alerts.append("🚨 **VERY HIGH LDL:** Aggressive lipid management needed.")

        factor_recommendations = []
        if score_breakdown.get('Blood Pressure', 0) >= 5:
            factor_recommendations.append("• Focus on strict BP control with medication and lifestyle.")
        if score_breakdown.get('Troponin', 0) > 0:
            factor_recommendations.append("• Cardiac enzymes elevated - immediate workup for Acute Coronary Syndrome (ACS) required.")
        if score_breakdown.get('Cholesterol', 0) >= 3:
            factor_recommendations.append("• Lipid management: Consult doctor regarding statin therapy due to high risk profile.")
        if age >= 50:
            factor_recommendations.append("• Age >50: Regular preventative cardiac screening is essential.")
        
        return base_rec, critical_alerts, factor_recommendations

# Initialize the Assessment System
assessment_engine = EnhancedCardioRiskAssessment()

@app.route("/")
def home():
    return render_template("dash.html")

@app.route("/brain_tumor.html")
def h():
    return render_template("brain_tumor.html")

@app.route("/heart_prediction.html")
def h2():
    return render_template("heart_prediction.html")

@app.route("/eye_glaucoma.html")
def h3():
    return render_template("eye_glaucoma.html")

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    
    # 1. Parse and Robustly Convert ALL Numeric Data (CRITICAL FIX - Type Safety)
    try:
        patient_data_casted = {
            'age': float(data.get('age', 40)),
            'sex': data.get('sex', 'Male'),
            'systolic_bp': float(data.get('systolic_bp', 120)),
            'diastolic_bp': float(data.get('diastolic_bp', 80)),
            'heart_rate': float(data.get('heart_rate', 70)),
            'troponin': float(data.get('troponin', 0.0)),
            'total_cholesterol': float(data.get('total_cholesterol', 180)),
            'ldl_cholesterol': float(data.get('ldl_cholesterol', 100)),
            'hdl_cholesterol': float(data.get('hdl_cholesterol', 50)),
            'triglycerides': float(data.get('triglycerides', 100)),
        }
    except Exception as e:
        return jsonify({'success': False, 'message': f'Invalid numeric data received: {e}'}), 400

    # 2. Calculate Metrics using the CASTED data
    try:
        risk_score, score_breakdown = assessment_engine.calculate_comprehensive_risk_score(patient_data_casted)
        risk_level_info = assessment_engine.get_risk_level(risk_score)
        
        base_rec, critical_alerts, factor_recs = assessment_engine.get_detailed_recommendations(
            risk_level_info, patient_data_casted, score_breakdown)
        
    except Exception as e:
         return jsonify({'success': False, 'message': f'Calculation failed: {e}'}), 500

    # 3. Create Visualization using the CASTED data
    fig = assessment_engine.create_comprehensive_visualization(
        risk_score, 
        risk_level_info, 
        patient_data_casted, 
        score_breakdown
    )
    
    # 4. Return JSON
    return jsonify({
        'success': True,
        'score': risk_score,
        'risk_level': risk_level_info['name'],
        'risk_color': risk_level_info['color'],
        'critical_alerts': critical_alerts,
        'recommendations': base_rec['actions'],
        'factor_recommendations': factor_recs,
        'graphJSON': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    })


# Load your trained model
model1 = tf.keras.models.load_model('glaucoma_model.keras')

# Define image size expected by the model
target_size = (256, 256)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # ✅ Fix: convert uploaded file to bytes and then to a readable stream
        img = load_img(io.BytesIO(file.read()), target_size=target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        prediction = model1.predict(img_array)[0][0]
        result = 'Glaucoma' if prediction > 0.5 else 'No Glaucoma'

        return jsonify({
            'prediction': result,
            'probability': float(prediction)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

class EpilepsyDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(EpilepsyDetectionModel, self).__init__()
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.efficientnet(x)

# --- Load the Trained Model ---
model_path = "epilepsy_detection_model.pth"
num_classes = 2
model = EpilepsyDetectionModel(num_classes=num_classes)

try:
    state_dict = torch.load(model_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("efficientnet."):
            new_key = k
        else:
            new_key = "efficientnet." + k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --- Image Transform ---
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Class Names ---
class_names = ['healthy', 'epilepsy']

# --- Severity Score Calculation ---
def calculate_severity_score(heatmap, model_confidence):
    if heatmap is None or heatmap.size == 0:
        return 0.0, "N/A"
    heatmap_threshold_value = np.percentile(heatmap, 70)
    activated_pixels = np.sum(heatmap > heatmap_threshold_value)
    total_pixels = heatmap.size
    area_ratio = activated_pixels / total_pixels if total_pixels > 0 else 0.0
    mean_intensity = np.mean(heatmap[heatmap > heatmap_threshold_value]) if activated_pixels > 0 else 0.0
    severity_score = (area_ratio * 0.4) + (mean_intensity * 0.3) + (model_confidence * 0.3)
    if severity_score < 0.3:
        severity_category = "Mild"
    elif 0.3 <= severity_score < 0.7:
        severity_category = "Moderate"
    else:
        severity_category = "Severe"
    return severity_score, severity_category

# --- Grad-CAM ---
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

    if class_names[predicted_class_idx] == 'healthy':
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

# --- Flask Routes ---
@app.route('/brain', methods=['POST'])
def brain():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
        img_tensor = inference_transforms(img)
        original_img = img.copy()

        with torch.no_grad():
            outputs = model(img_tensor.unsqueeze(0))
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(outputs, dim=1).item()
            model_confidence = probabilities[0, predicted_class_idx].item()
            predicted_class_name = class_names[predicted_class_idx]

        heatmap_base64 = None
        severity_score, severity_category = 0.0, "N/A"

        if predicted_class_name == 'epilepsy':
            heatmap, _, _ = generate_cam_heatmap(model, img_tensor, predicted_class_idx)
            if heatmap is not None:
                severity_score, severity_category = calculate_severity_score(heatmap, model_confidence)
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


