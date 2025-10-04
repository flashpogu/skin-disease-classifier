import modal

app = modal.App("skin-disease-classifier")

vol = modal.Volume.from_name("ham10000-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/root/skin-disease-classifier")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60*60*4,
    volumes={"/root/data": vol},
)
def train_on_modal():
    import sys
    import os
    sys.path.append("/root/skin-disease-classifier")

    # Check the volume contents
    print("Volume root contents:", os.listdir("/root/data"))
    if os.path.exists("/root/data/HAM10000_images"):
        print("First 5 images:", os.listdir("/root/data/HAM10000_images")[:5])
    else:
        print("Images folder not found!")

    metadata_path = "/root/data/root/data/HAM10000_metadata.csv"
    images_path = "/root/data/root/data/HAM10000_images"

    import train
    train.train(
        metadata_csv=metadata_path,
        images_dir=images_path
    )
