import os
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import WeightedRandomSampler

from dataset import get_splits, LABELS
from models.resnet import get_model


def make_balanced_sampler(train_loader):
    targets = train_loader.dataset.df['dx'].map(LABELS).values
    class_count = torch.bincount(torch.tensor(targets))
    weight = 1.0 / class_count.float()
    sample_weights = torch.tensor([weight[t] for t in targets])
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def validate(model, val_loader, device):
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outs = model(imgs).cpu()
            preds.append(outs.softmax(1))
            labs.append(labels)
    y_prob = torch.cat(preds).numpy()
    y_true = torch.cat(labs).numpy()
    y_pred = y_prob.argmax(1)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    except:
        auc = 0.0
    acc = 100. * (y_pred == y_true).sum() / len(y_true)
    return auc, acc


def save_best(model, auc):
    path = "/mnt/data/HAM10000_images/best_model.pth"
    torch.save(model.state_dict(), path)
    print(f"✅ Saved best model (AUC {auc:.4f}) → {path}")


def train_phase(model, train_loader, val_loader, device, optimizer, scheduler,
                loss_fn, epochs, label, patience=10):
    best_auc = -1
    no_imp = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = loss_fn(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, pred = out.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step(epoch)
        train_acc = 100. * correct / total
        val_auc, val_acc = validate(model, val_loader, device)

        print(f"{label} | Epoch {epoch} | Loss {total_loss/len(train_loader):.4f} "
              f"| TrainAcc {train_acc:.2f}% | ValAUC {val_auc:.4f} | ValAcc {val_acc:.2f}%")

        if val_auc > best_auc:
            best_auc, no_imp = val_auc, 0
            save_best(model, val_auc)
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"⏹ {label}: Early stop at {epoch}")
                break


def train(metadata_csv="HAM10000_metadata.csv", images_dir="HAM10000_images/",
          batch_size=32, epochs=50, patience=7):

    print("Loading data...")
    train_loader, val_loader, _ = get_splits(
        metadata_csv, images_dir, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Balanced sampler to improve minority class recall
    sampler = make_balanced_sampler(train_loader)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    model = get_model().to(device)
    loss_fn = nn.CrossEntropyLoss()

    # -------- 🔥 Phase 1: Train only classifier head --------
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    optim1 = torch.optim.Adam(model.fc.parameters(), lr=3e-3)
    sched1 = CosineAnnealingWarmRestarts(optim1, T_0=8)
    print("\n========== Phase 1: Head Only ==========")
    train_phase(model, train_loader, val_loader, device, optim1, sched1, loss_fn,
                epochs=15, label="Phase 1")

    # -------- 🔥 Phase 2: Unfreeze deeper layers --------
    for p in model.layer3.parameters():
        p.requires_grad = True
    for p in model.layer4.parameters():
        p.requires_grad = True
    optim2 = torch.optim.Adam([
        {"params": model.layer3.parameters(), "lr": 1e-4},
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 1e-3},
    ])
    sched2 = CosineAnnealingWarmRestarts(optim2, T_0=12)
    print("\n========== Phase 2: Last Blocks ==========")
    train_phase(model, train_loader, val_loader, device, optim2, sched2, loss_fn,
                epochs=25, label="Phase 2")

    # -------- 🔥 Phase 3: Full fine-tuning --------
    for p in model.parameters():
        p.requires_grad = True
    optim3 = torch.optim.Adam(model.parameters(), lr=1e-5)
    sched3 = CosineAnnealingWarmRestarts(optim3, T_0=20)
    print("\n========== Phase 3: Full FT ==========")
    train_phase(model, train_loader, val_loader, device, optim3, sched3, loss_fn,
                epochs=25, label="Phase 3")

    print("\n✅ Training fully completed with advanced strategy!\n")


if __name__ == "__main__":
    train()
