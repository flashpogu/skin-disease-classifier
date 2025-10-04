import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from dataset import get_splits, LABELS
from model import get_model
from torch.optim.lr_scheduler import CosineAnnealingLR


def get_loss_fn(train_df, device):
    # Count samples per class
    class_counts = train_df['dx'].value_counts().to_dict()

    # Order counts according to LABELS mapping (0..6)
    counts_list = [class_counts[cls] for cls in LABELS.keys()]

    # Compute weights: inverse frequency
    weights = 1.0 / torch.tensor(counts_list, dtype=torch.float)

    # Normalize so weights aren’t too large
    weights = weights / weights.sum() * len(LABELS)

    return nn.CrossEntropyLoss(weight=weights.to(device))


def train(metadata_csv="HAM10000_metadata.csv",
          images_dir="HAM10000_images/",
          batch_size=32,
          epochs=50,
          patience=7):
    # dataset
    train_loader, val_loader, test_loader = get_splits(
        metadata_csv,
        images_dir,
        batch_size=batch_size
    )

    # device + model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = get_loss_fn(train_loader.dataset.df, device)

    model = get_model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = -1.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # ---- Validation ----
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                out = model(imgs).cpu()
                all_preds.append(out.softmax(1))
                all_labels.append(labels)

        y_prob = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        y_pred = y_prob.argmax(1)

        try:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except Exception:
            auc = None

        print(f"Epoch {epoch} | loss {total_loss:.3f} | val_auc {auc}")

        # detailed metrics
        target_names = [cls for cls, _ in sorted(
            LABELS.items(), key=lambda x: x[1])]
        print(classification_report(
            y_true, y_pred, target_names=target_names, zero_division=0))

        # Optional: print confusion matrix
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

        # early stopping check
        if auc is not None and auc > best_val_auc:
            best_val_auc = auc
            epochs_no_improve = 0
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), os.path.join(
                images_dir, "best_model.pth"))
            print(
                f"✅ Saved best model at epoch {epoch} with val_auc {auc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("⏹ Early stopping triggered!")
                break


if __name__ == "__main__":
    train()
