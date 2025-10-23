# 🧠 Skin Disease Classifier – Explainable Medical AI

A full-stack medical imaging application that detects 7 types of skin lesions using Deep Learning and provides **Grad-CAM explainability** along with a **clinical-style PDF diagnostic report**.

Powered by:
✅ PyTorch  
✅ FastAPI  
✅ Modal Cloud (GPU Inference)  
✅ Next.js Frontend  
✅ ReportLab PDF Generator  

---

## 🚀 Features

| Feature | Status |
|--------|:-----:|
| AI classification (7 skin disease classes) | ✅ |
| Grad-CAM heatmap explainability | ✅ |
| Cloud GPU inference via Modal | ✅ |
| PDF diagnostic report generator | ✅ |
| Probability scores for all classes | ✅ |
| Modern UI with heatmap overlay | ✅ |


---

## 🧬 Supported Skin Disease Classes (HAM10000)

| Code | Full Medical Name | Severity |
|------|------------------|:-------:|
| **mel** | Melanoma | 🔥 High |
| **akiec** | Actinic Keratoses | ⚠ Medium |
| **bcc** | Basal Cell Carcinoma | ⚠ Medium |
| **vasc** | Vascular Lesions | Low |
| **df** | Dermatofibroma | Low |
| **bkl** | Benign Keratosis | Low |
| **nv** | Melanocytic Nevi (Mole) | Low |

---

## 🏗️ System Architecture


---

## 🖥️ UI Overview

✅ Upload Skin Image  
✅ View Original & Grad-CAM Heatmap  
✅ Download medical PDF report  
✅ Confidence visualization

---

## ⚙️ Tech Stack

| Layer | Technology |
|------|------------|
| Backend API | FastAPI, Modal |
| AI Model | PyTorch, ResNet50 |
| Explainability | Grad-CAM |
| PDF Generation | ReportLab |
| Frontend UI | React, Next.js 14, ShadCN |
| Dataset | HAM10000 |
| Deployment | Modal Cloud (GPU) |

---

## 📁 Project Structure

📦 skin-disease-classifier
┣ 📂 models/ (ResNet model files)
┣ 📂 HAM10000_images/ (image dataset & checkpoints)
┣ 📂 classifier-frontend/ (Next.js UI)
┣ 📜 train.py (3-phase fine-tuning)
┣ 📜 main.py (API + Grad-CAM + PDF endpoints)
┣ 📜 dataset.py (data loader with augmentation)
┣ 📜 gradcam.py (heatmap generation)
┣ 📜 requirements.txt


---

## 🔥 Training Strategy

We fine-tune **ResNet-50** using:

✅ Phase 1 — Train Head  
✅ Phase 2 — Train last residual blocks  
✅ Phase 3 — Full network fine-tuning  
✅ Early Stopping + Weighted Loss for Class Imbalance  

Final Validation AUC: **0.97+** 🚀  

---

## 🧪 API Usage

### 🔍 Inference Request
```http
POST /inference
Content-Type: application/json

{
  "image_data": "<base64-encoded image>"
}
{
  "predicted_label": "nv",
  "heatmap": "<base64>",
  "probabilities": [...]
}

