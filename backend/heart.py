import json
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
import numpy as np 

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
    return render_template("heart_prediction.html")

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

if __name__ == '__main__':
    print("Starting Enhanced Cardiovascular Risk Assessment API (V9 - Final Viz Fix) on Port 5000...")
    # IMPORTANT: Use use_reloader=False if you see caching issues after repeated saves.
    app.run(debug=True, port=5000)