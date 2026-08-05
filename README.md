# Prognosys: A Unified Multi-Modal Framework for Heterogeneous Disease Prediction and Context-Aware Patient Education

[![IEEE Xplore Paper](https://img.shields.io/badge/IEEE%20Xplore-10.1109%2FICBSII69710.2026.11479259-blue)](https://ieeexplore.ieee.org/document/11479259)
[![Conference](https://img.shields.io/badge/ICBSII-2026-brightgreen)](https://ieeexplore.ieee.org/document/11479259)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Flask-Backend-orange.svg)](https://flask.palletsprojects.org/)

---

## 📌 Paper Details & Citation

**Publication:** 2026 Twelfth International Conference on Bio Signals, Images, and Instrumentation (ICBSII)  
**Location:** Chennai, India  
**IEEE Document ID:** [11479259](https://ieeexplore.ieee.org/document/11479259)  
**DOI:** [10.1109/ICBSII69710.2026.11479259](https://doi.org/10.1109/ICBSII69710.2026.11479259)  

**Authors:**
* **R. Premkumar** - Department of Biomedical Engineering, Rajalakshmi Engineering College, Chennai, India
* **Yugesh Elumalai H** - Department of Biomedical Engineering, Rajalakshmi Engineering College, Chennai, India
* **Rithesh Prayas** - Department of Biomedical Engineering, Rajalakshmi Engineering College, Chennai, India
* **Nithish S** - Department of Biomedical Engineering, Rajalakshmi Engineering College, Chennai, India

---

## 📖 Abstract

The contemporary digital health landscape is characterized by fragmentation, forcing patients to navigate a disconnected ecosystem of single-purpose applications. This "siloed" architectural paradigm imposes a significant cognitive burden on users and impedes the transition from reactive to proactive care.

**Prognosys** is a unified, cross-platform framework designed to democratize access to predictive healthcare. Built with a modular API routing architecture, it services both a responsive web interface and a native mobile application. The system integrates three heterogeneous diagnostic pipelines:
1. **Hybrid Cardiovascular Risk Engine**: Combines Random Forest with a clinical rule-based scoring algorithm.
2. **Brain Structural Anomaly & Epilepsy Pipeline**: Utilizes an **EfficientNet-B0 Convolutional Neural Network (CNN)** for detecting structural brain anomalies (tumors) as predictive biomarkers with Grad-CAM heatmap visualization.
3. **Retinal Fundus Glaucoma Screening**: Employs a **VGG16 / Keras CNN** model for glaucoma prediction from retinal images.

Furthermore, Prognosys incorporates a **Generative AI** module, encased in a safety wrapper, to deliver context-aware patient education. Usability validation via the System Usability Scale (SUS) yielded a score of **85.3**, confirming the user satisfaction and efficacy of the modern **Glassmorphism** UI interface.

---

## 📊 Experimental Results & Diagnostic Accuracy

| Diagnostic Module | Primary Model / Pipeline | Targeted Condition | Accuracy |
| :--- | :--- | :--- | :---: |
| 🧠 **Brain Module** | EfficientNet-B0 (PyTorch) + Grad-CAM | Structural Brain Anomalies / Epilepsy | **94.0%** |
| 👁️ **Eye Module** | VGG16 / Keras CNN | Retinal Fundus Glaucoma | **92.1%** |
| ❤️ **Heart Module** | Hybrid Random Forest + Clinical Rule Engine | Cardiovascular Disease Risk | **88.5%** |

---

## 🌟 Key Features

* **Multi-Modal Diagnostic Support**: Simultaneous analysis for Cardiovascular, Neurological (Brain MRI), and Ophthalmic (Retinal Fundus) data.
* **Explainable AI (XAI)**: Includes **Grad-CAM (Gradient-weighted Class Activation Mapping)** for visual localization of detected brain lesions/anomalies directly on input MRI scans.
* **Generative AI Patient Education**: Provides personalized, safety-wrapped medical context explanations based on diagnostic outputs.
* **Glassmorphism Web Dashboard**: Responsive, highly intuitive UI designed for high engagement (SUS Score: 85.3).
* **Flask Microservice Architecture**: Decoupled RESTful API routing servicing web and mobile client frontends.

---

## 📁 Repository Structure

```
final_yr_project_webpage/
│
├── backend/
│   ├── app.py                         # Main Flask application server & API routes
│   ├── epilepsy_detection_model.pth    # PyTorch EfficientNet-B0 trained model weights
│   ├── glaucoma_model.keras           # Keras VGG16 trained model weights (optional)
│   └── __init__.py                    # Backend package initializer
│
├── templates/
│   ├── dash.html                      # Glassmorphism main dashboard UI
│   ├── index.html                     # Landing homepage
│   ├── brain_tumor.html               # Brain MRI & Epilepsy analysis portal
│   ├── eye_glaucoma.html              # Retinal Fundus Glaucoma diagnostic portal
│   ├── heart_prediction.html          # Cardiovascular risk engine portal
│   └── *.mp4                          # Demonstration & diagnostic preview videos
│
├── requirements.txt                   # Python dependencies list
├── runtime.txt                        # Runtime environment configuration
└── README.md                          # Project & Paper documentation
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.9 or higher
* `pip` package manager
* Virtual environment (recommended)

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yugi252179/final_yr_project_webpage.git
   cd final_yr_project_webpage
   ```

2. **Set Up Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Core requirements include `flask`, `flask-cors`, `torch`, `torchvision`, `tensorflow`, `opencv-python`, `pillow`, `plotly`, `numpy`)*

4. **Launch the Application:**
   ```bash
   python backend/app.py
   ```

5. **Access Web Interface:**
   Open your browser and navigate to `http://localhost:5000` (or `http://127.0.0.1:5000`).

---

## 🔌 API Endpoints Summary

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/` | `GET` | Main Glassmorphism Dashboard Page (`dash.html`) |
| `/brain_tumor.html` | `GET` | Brain Tumor & Epilepsy diagnostic interface |
| `/heart_prediction.html` | `GET` | Cardiovascular Risk calculation interface |
| `/eye_glaucoma.html` | `GET` | Glaucoma screening interface |
| `/predict` | `POST` | Glaucoma diagnostic inference (Accepts fundus image file) |
| `/brain` | `POST` | Epilepsy & MRI anomaly inference + Grad-CAM heatmap generation |

---

## 📜 Citation (BibTeX)

If you find this work useful or reference Prognosys in your research, please cite our paper:

```bibtex
@INPROCEEDINGS{11479259,
  author={Premkumar, R. and H, Yugesh Elumalai and Prayas, Rithesh and S, Nithish},
  booktitle={2026 Twelfth International Conference on Bio Signals, Images, and Instrumentation (ICBSII)}, 
  title={Prognosys: A Unified Multi-Modal Framework for Heterogeneous Disease Prediction and Context-Aware Patient Education}, 
  year={2026},
  pages={1-5},
  keywords={Multi-Modal AI, Mobile Health (mHealth), EfficientNet, VGG16, Clinical Decision Support, Medical Image Analysis},
  doi={10.1109/ICBSII69710.2026.11479259}
}
```

---


