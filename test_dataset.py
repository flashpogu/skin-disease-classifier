from dataset import HAM10000Dataset, LABELS

# point to your dataset paths
metadata_csv = "HAM10000_metadata.csv"
images_dir = "HAM10000_images"

# get train/val/test loaders
train_loader, val_loader, test_loader = HAM10000Dataset.get_splits(
    metadata_csv, images_dir, batch_size=4)

print("Dataset splits created")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# take one batch and inspect
images, labels = next(iter(train_loader))

print("Batch images shape:", images.shape)   # [batch_size, 3, 224, 224]
print("Batch labels:", labels)               # tensor([..])

# check label mapping
inv_labels = {v: k for k, v in LABELS.items()}
print("Decoded labels:", [inv_labels[l.item()] for l in labels])
