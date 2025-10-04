import base64
import json

# Path to your image
image_path = "HAM10000_images/ISIC_0034317.jpg"

# Path to payload.json
payload_path = "payload.json"

# Read image as bytes
with open(image_path, "rb") as f:
    img_bytes = f.read()

# Encode to base64 string
img_b64 = base64.b64encode(img_bytes).decode("utf-8")

# Create JSON structure
payload = {"image_data": img_b64}

# Write into payload.json (overwrite if exists)
with open(payload_path, "w") as f:
    json.dump(payload, f)

print(f"✅ Wrote base64 of {image_path} into {payload_path}")
