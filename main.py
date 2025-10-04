import modal
import torch
from PIL import Image
from torchvision import transforms
import io
import base64
import sys
from pydantic import BaseModel

app = modal.App("skin-disease-classifier")

# Volume for model and data
vol = modal.Volume.from_name("ham10000-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/root/skin-disease-classifier")
)

LABELS = {'nv': 0, 'mel': 1, 'bkl': 2,
          'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}
idx2label = {v: k for k, v in LABELS.items()}

# transforms (same as validation)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class ImageRequest(BaseModel):
    image_data: str  # base64 encoded image


@app.cls(image=image, gpu="A10G", volumes={"/root/data": vol}, scaledown_window=15)
class SkinClassifier:
    @modal.enter()
    def load_model(self):
        """Load model once when the container starts"""
        sys.path.append("/root/skin-disease-classifier")
        from model import get_model

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "/root/data/root/data/HAM10000_images/best_model.pth"

        self.model = get_model().to(self.device)
        self.model.load_state_dict(torch.load(
            self.model_path, map_location=self.device))
        self.model.eval()
        print("✅ Skin disease model loaded")

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: ImageRequest):
        # decode base64 image
        image_bytes = base64.b64decode(request.image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = transform(img).unsqueeze(0).to(self.device)

        # inference
        with torch.no_grad():
            outputs = self.model(img)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

        response = {
            "predicted_label": idx2label[pred_class],
            "probabilities": probs.cpu().numpy().tolist()
        }
        return response


# local entrypoint for testing
@app.local_entrypoint()
def main():
    server = SkinClassifier()

    # example: read local image and send as base64
    with open("HAM10000_images/ISIC_0024306.jpg", "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    payload = {"image_data": img_b64}

    # call the FastAPI endpoint
    url = server.inference.get_web_url()
    import requests
    response = requests.post(url, json=payload)
    response.raise_for_status()

    result = response.json()
    print(result)
