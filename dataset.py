import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# Disease labels mapping
LABELS = {
    'akiec': 0,  # Actinic keratoses
    'bcc': 1,    # Basal cell carcinoma
    'bkl': 2,    # Benign keratosis
    'df': 3,     # Dermatofibroma
    'mel': 4,    # Melanoma
    'nv': 5,     # Melanocytic nevi
    'vasc': 6    # Vascular lesions
}


class HAM10000Dataset(Dataset):
    """
    Dataset class for HAM10000 skin lesion images.
    """

    def __init__(self, df, images_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['image_id']
        label = LABELS[row['dx']]

        # Try .jpg first, then .png
        img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.images_dir, f"{img_id}.png")

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(is_training=True):
    """
    Get data augmentation transforms for training and validation.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def get_splits(metadata_csv, images_dir, batch_size=32, val_split=0.15, test_split=0.15):
    """
    Load data and create train/val/test DataLoaders.

    Args:
        metadata_csv: Path to metadata CSV file
        images_dir: Directory containing images
        batch_size: Batch size for DataLoaders
        val_split: Validation set proportion
        test_split: Test set proportion

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load metadata
    df = pd.read_csv(metadata_csv)

    # Split: train / (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_split + test_split),
        stratify=df['dx'],
        random_state=42
    )

    # Split: val / test
    val_ratio = val_split / (val_split + test_split)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio),
        stratify=temp_df['dx'],
        random_state=42
    )

    print(f"Dataset splits:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")

    # Create datasets
    train_dataset = HAM10000Dataset(
        train_df, images_dir, transform=get_transforms(True))
    val_dataset = HAM10000Dataset(
        val_df, images_dir, transform=get_transforms(False))
    test_dataset = HAM10000Dataset(
        test_df, images_dir, transform=get_transforms(False))

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
