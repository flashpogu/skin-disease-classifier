import torch
from models.efnet import get_model
from PIL import Image
from torchvision import transforms
import os

# class labels
LABELS = {'nv': 0, 'mel': 1, 'bkl': 2,
          'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}
idx2label = {v: k for k, v in LABELS.items()}

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = get_model().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# image transforms (same as training/validation)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    return idx2label[pred_class], probs.cpu().numpy()


# inference on folder
# folder with test images
test_dir = "C:\\Skin-Disease-Classifier\\HAM10000_images\\ISIC_0024402.jpg"
label, probs = predict_image(test_dir)
print(f"{test_dir}: Predicted -> {label}, Probabilities -> {probs}")
