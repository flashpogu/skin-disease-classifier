import modal
import torch
from PIL import Image
from torchvision import transforms
import io
import base64
import sys
from pydantic import BaseModel
from gradcam import GradCAM
import numpy as np
import cv2
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

app = modal.App("skin-disease-classifier")

vol = modal.Volume.from_name("ham10000-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/root")
)

LABELS = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2,
    'df': 3,
    'mel': 4,
    'nv': 5,
    'vasc': 6
}
idx2label = {v: k for k, v in LABELS.items()}

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class ImageRequest(BaseModel):
    image_data: str


@app.cls(image=image, gpu="A10G", volumes={"/mnt/data": vol}, scaledown_window=15)
class SkinClassifier:

    @modal.enter()
    def load_model(self):
        sys.path.append("/root")
        from models.resnet import get_model

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "/mnt/data/HAM10000_images/best_model.pth"

        print("Files in /root:", os.listdir("/root"))

        self.model = get_model().to(self.device)
        self.model.load_state_dict(torch.load(
            self.model_path, map_location=self.device))
        self.model.eval()

        target_layer = self.model.layer4[-1]
        self.gradcam = GradCAM(self.model, target_layer)

        print("✅ Model + GradCAM Ready")

    def run_inference(self, img):
        processed = transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(processed)
            probs = torch.softmax(outputs, dim=1).cpu().numpy().tolist()
            pred_class = int(np.argmax(probs))

        heatmap, _ = self.gradcam.generate(processed, pred_class)

        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        heatmap = cv2.GaussianBlur(heatmap, (7, 7), 0)

        original = cv2.cvtColor(
            np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(original, 0.35, heatmap, 0.65, 0)

        _, buffer = cv2.imencode(".jpg", overlay)
        heatmap_b64 = base64.b64encode(buffer).decode("utf-8")

        return pred_class, probs, heatmap_b64

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: ImageRequest):
        img = Image.open(io.BytesIO(base64.b64decode(
            request.image_data))).convert("RGB")
        pred_class, probs, heatmap_b64 = self.run_inference(img)

        return {
            "predicted_label": idx2label[pred_class],
            "probabilities": probs,
            "heatmap": heatmap_b64
        }

    @modal.fastapi_endpoint(method="POST")
    def report(self, request: ImageRequest):
        img = Image.open(io.BytesIO(base64.b64decode(
            request.image_data))).convert("RGB")
        pred_class, probs, heatmap_b64 = self.run_inference(img)

        diagnosis = idx2label[pred_class]
        confidence = max(probs[0])

        # Save original image
        original = img.resize((256, 256))
        original_path = "/tmp/original.jpg"
        original.save(original_path)

        # Save heatmap overlay properly
        heatmap_bytes = base64.b64decode(heatmap_b64)
        heatmap_np = np.frombuffer(heatmap_bytes, np.uint8)
        heatmap_img = cv2.imdecode(heatmap_np, cv2.IMREAD_COLOR)
        heatmap_path = "/tmp/heatmap.jpg"
        cv2.imwrite(heatmap_path, heatmap_img)

        # Medical disease descriptions
        disease_info = {
            "mel": "High-risk skin cancer. Seek urgent dermatologist evaluation.",
            "nv": "Common mole. Low concern unless changing in size or color.",
            "bkl": "Tan/rough benign lesion, often age-related and harmless.",
            "bcc": "Common slow-growing skin cancer. Early treatment recommended.",
            "akiec": "Precancerous lesion due to sun exposure. Monitoring advised.",
            "vasc": "Blood-vessel related lesion. Usually low concern.",
            "df": "Benign firm bump on the skin. Typically harmless."
        }

        risk_message = disease_info.get(
            diagnosis, "Consult a dermatologist for confirmation.")

        # Risk severity coloring
        risk_level = "Low" if confidence > 0.85 else "Medium" if confidence > 0.6 else "High"

        # Create PDF
        pdf_path = "/tmp/report.pdf"
        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4

        # Header
        c.setFont("Helvetica-Bold", 20)
        c.drawString(40, height - 40, "Skin Lesion AI Diagnostic Report")

        # Diagnosis Summary
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, height - 90, f"Diagnosis: {diagnosis.upper()}")
        c.setFont("Helvetica", 12)
        c.drawString(40, height - 110, f"Condition: {disease_info[diagnosis]}")
        c.drawString(40, height - 130, f"Confidence: {confidence*100:.2f}%")
        c.drawString(40, height - 150, f"Risk Level: {risk_level}")

        # Insert original & heatmap
        IMAGE_Y = height - 460  # ✅ Move images lower

        c.drawImage(original_path, 40, IMAGE_Y, width=256, height=256)
        c.drawImage(heatmap_path, 320, IMAGE_Y, width=256, height=256)

        # Footer
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(
            40, 40, "Developed by Rahul Kumar • This is not a medical diagnosis.")

        c.showPage()
        c.save()

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        return {"pdf_report": base64.b64encode(pdf_bytes).decode("utf-8")}


@app.local_entrypoint()
def main():
    server = SkinClassifier()
    with open("HAM10000_images/ISIC_0024306.jpg", "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    url = server.inference.get_web_url()
    import requests
    res = requests.post(url, json={"image_data": b64})
    print(res.json())
