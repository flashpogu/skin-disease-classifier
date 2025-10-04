import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

LABELS = {'nv': 0, 'mel': 1, 'bkl': 2,
          'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}


class HAM10000Dataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row['image_id'] + '.jpg')
        img = Image.open(img_path).convert('RGB')
        label = LABELS[row['dx']]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_splits(metadata_csv, images_dir, batch_size=32, seed=42):
    df = pd.read_csv(metadata_csv)
    df = df.rename(columns={'image_id': 'image_id', 'dx': 'dx'})

    # Split train/test (70/30 stratified)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    X = df['image_id']
    y = df['dx']
    train_idx, test_idx = next(sss.split(X, y))
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    # split test --> val and test
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx2 = next(sss2.split(test_df['image_id'], test_df['dx']))
    val_df = test_df.iloc[val_idx]
    test_df = test_df.iloc[test_idx2]

    # transforms
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ====== Weighted Sampler ======
    class_counts = train_df['dx'].value_counts().to_dict()
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[row['dx']] for _, row in train_df.iterrows()]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    # Dataloaders
    train_loader = DataLoader(
        HAM10000Dataset(train_df, images_dir, train_tf),
        batch_size=batch_size,
        sampler=sampler,   # <-- sampler instead of shuffle=True
        num_workers=4
    )
    val_loader = DataLoader(
        HAM10000Dataset(val_df, images_dir, val_tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        HAM10000Dataset(test_df, images_dir, val_tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader, test_loader
