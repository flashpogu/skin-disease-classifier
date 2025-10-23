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
    volumes={"/mnt/data": vol},
    timeout=300,
)
def check_volume():
    """
    Diagnostic function to inspect the volume structure.
    Run with: modal run modal_train.py::check_volume
    """
    import os

    def list_files(path, depth=0, max_depth=3):
        if depth > max_depth:
            return
        try:
            items = sorted(os.listdir(path))
            for item in items[:15]:  # Show first 15 items
                item_path = os.path.join(path, item)
                indent = "  " * depth
                if os.path.isdir(item_path):
                    print(f"{indent}📁 {item}/")
                    list_files(item_path, depth + 1, max_depth)
                else:
                    size = os.path.getsize(item_path)
                    print(f"{indent}📄 {item} ({size:,} bytes)")
        except (PermissionError, OSError) as e:
            print(f"{indent}❌ Error: {e}")

    print("=" * 80)
    print("VOLUME STRUCTURE INSPECTION")
    print("=" * 80)
    list_files("/mnt/data")
    print("=" * 80)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 4,  # 4 hours
    volumes={"/mnt/data": vol},
)
def train_on_modal():
    """
    Main training function.
    Run with: modal run modal_train.py::train_on_modal
    """
    import sys
    import os

    # Add project to Python path
    sys.path.append("/root/skin-disease-classifier")

    print("=" * 80)
    print("CHECKING VOLUME CONTENTS")
    print("=" * 80)

    # Check volume structure
    print("Volume root contents:", os.listdir("/mnt/data"))

    # Determine correct paths based on actual structure
    if os.path.exists("/mnt/data/root/data/HAM10000_images"):
        print("✓ Found nested structure (/mnt/data/root/data/...)")
        metadata_path = "/mnt/data/root/data/HAM10000_metadata.csv"
        images_path = "/mnt/data/root/data/HAM10000_images"
    elif os.path.exists("/mnt/data/HAM10000_images"):
        print("✓ Found flat structure (/mnt/data/...)")
        metadata_path = "/mnt/data/HAM10000_metadata.csv"
        images_path = "/mnt/data/HAM10000_images"
    else:
        print("❌ ERROR: Could not find HAM10000_images!")
        print("Available items in /mnt/data:")
        for item in os.listdir("/mnt/data"):
            print(f"  - {item}")
        raise FileNotFoundError(
            "HAM10000_images directory not found in volume")

    # Verify files exist
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata CSV not found at {metadata_path}")

    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Images directory not found at {images_path}")

    # Count images
    num_images = len([f for f in os.listdir(images_path)
                     if f.endswith(('.jpg', '.png'))])
    print(f"✓ Found {num_images} images in {images_path}")

    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    # Import and run training
    import train
    train.train(
        metadata_csv=metadata_path,
        images_dir=images_path,
        batch_size=32,
        epochs=50,
        patience=7
    )

    # Commit changes to volume (save checkpoints)
    vol.commit()
    print("✓ Volume changes committed (checkpoints saved)")


@app.local_entrypoint()
def main():
    """
    Default entrypoint - runs training.
    Run with: modal run modal_train.py
    """
    train_on_modal.remote()
