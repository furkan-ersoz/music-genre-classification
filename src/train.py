"""
Training loop for music genre classification.
Supports early stopping, LR scheduling, and checkpoint saving.
Called by run_all.py — not intended to be run directly.
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from pathlib import Path


def train_model(model: nn.Module,
                train_loader,
                val_loader,
                config: dict,
                run_id: str,
                device: torch.device) -> dict:
    """
    Train model and return best checkpoint info.

    Returns:
        {
            "best_val_accuracy": float,
            "best_epoch":        int,
            "train_time_sec":    float,
            "checkpoint_path":   str,
        }
    """
    epochs    = config.get("epochs", 50)
    lr        = config.get("lr", 1e-3)
    patience  = config.get("early_stopping_patience", 10)
    scheduler_type = config.get("scheduler", "plateau")

    checkpoint_dir = Path("results") / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                      patience=3, factor=0.5)

    best_val_acc  = 0.0
    best_epoch    = 0
    no_improve    = 0
    history       = {"train_loss": [], "val_loss": [],
                     "train_acc":  [], "val_acc":  []}

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += images.size(0)

        train_loss /= train_total
        train_acc   = train_correct / train_total

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs  = model(images)
                loss     = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds     = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += images.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total

        # ── Scheduler step ────────────────────────────────────────────────────
        if scheduler_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_acc)

        # ── History ───────────────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:03d}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            no_improve   = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    train_time = time.time() - start_time

    # Save training curves for evaluate.py
    import json
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f)

    print(f"\nBest val_acc={best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Training time: {train_time:.1f}s")
    print(f"Checkpoint: {checkpoint_path}")

    return {
        "best_val_accuracy": best_val_acc,
        "best_epoch":        best_epoch,
        "train_time_sec":    round(train_time, 1),
        "checkpoint_path":   str(checkpoint_path),
        "history":           history,
    }
