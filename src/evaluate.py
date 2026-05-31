"""
Evaluation and results reporting for music genre classification.
Appends one row per (model, test_dataset) to results/results.csv.
Saves confusion matrix and training curves under results/<run_id>/.
"""

import os
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix)

RESULTS_CSV = Path("results") / "results.csv"
CSV_COLUMNS = [
    "run_id", "model", "train_on", "test_on",
    "epoch", "accuracy", "f1_macro", "f1_weighted",
    "precision_macro", "recall_macro", "loss",
    "train_time_sec", "best_val_accuracy",
    "config_path", "notes",
]


def _ensure_csv_header():
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_CSV.exists():
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def evaluate_model(model: nn.Module,
                   test_loader,
                   label_encoder,
                   device: torch.device) -> dict:
    """Run inference, return metrics dict."""
    model.eval()
    criterion   = nn.CrossEntropyLoss()
    all_preds   = []
    all_labels  = []
    total_loss  = 0.0
    total_count = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs     = model(images)
            loss        = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_count += images.size(0)
            preds       = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_count

    return {
        "accuracy":        round(accuracy_score(all_labels, all_preds), 4),
        "f1_macro":        round(f1_score(all_labels, all_preds, average="macro",  zero_division=0), 4),
        "f1_weighted":     round(f1_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
        "precision_macro": round(precision_score(all_labels, all_preds, average="macro", zero_division=0), 4),
        "recall_macro":    round(recall_score(all_labels, all_preds, average="macro", zero_division=0), 4),
        "loss":            round(avg_loss, 4),
        "all_preds":       all_preds,
        "all_labels":      all_labels,
    }


def save_confusion_matrix(all_labels, all_preds, label_encoder,
                          out_path: str, title: str = "") -> None:
    class_names = label_encoder.classes_
    cm = confusion_matrix(all_labels, all_preds,
                          labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion Matrix")
    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"  Confusion matrix saved: {out_path}")


def save_training_curves(history: dict, out_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["val_loss"],   label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["val_acc"],   label="val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"  Training curves saved: {out_path}")


def record_results(run_id: str,
                   model_name: str,
                   train_on: list,
                   test_on: str,
                   metrics: dict,
                   train_info: dict,
                   config_path: str,
                   notes: str = "") -> None:
    """Append one row to results/results.csv."""
    _ensure_csv_header()
    row = {
        "run_id":            run_id,
        "model":             model_name,
        "train_on":          "+".join(train_on),
        "test_on":           test_on,
        "epoch":             train_info.get("best_epoch", ""),
        "accuracy":          metrics["accuracy"],
        "f1_macro":          metrics["f1_macro"],
        "f1_weighted":       metrics["f1_weighted"],
        "precision_macro":   metrics["precision_macro"],
        "recall_macro":      metrics["recall_macro"],
        "loss":              metrics["loss"],
        "train_time_sec":    train_info.get("train_time_sec", ""),
        "best_val_accuracy": train_info.get("best_val_accuracy", ""),
        "config_path":       config_path,
        "notes":             notes,
    }
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)
    print(f"  Result recorded → results/results.csv")


def run_evaluation(model: nn.Module,
                   test_loaders: dict,
                   label_encoder,
                   train_info: dict,
                   run_id: str,
                   model_name: str,
                   train_on: list,
                   config_path: str,
                   config: dict,
                   device: torch.device) -> None:
    """
    Full evaluation pipeline:
        - Evaluate on each test dataset
        - Save confusion matrices
        - Save training curves
        - Record to results.csv
    """
    out_dir = Path("results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training curves (once per run)
    if "history" in train_info:
        save_training_curves(
            train_info["history"],
            str(out_dir / "training_curves.png")
        )

    # Per-dataset evaluation
    for dataset_name, test_loader in test_loaders.items():
        print(f"\n── Evaluating on {dataset_name} ──")
        metrics = evaluate_model(model, test_loader, label_encoder, device)

        print(f"  accuracy={metrics['accuracy']}  "
              f"f1_macro={metrics['f1_macro']}  "
              f"f1_weighted={metrics['f1_weighted']}")

        save_confusion_matrix(
            metrics["all_labels"],
            metrics["all_preds"],
            label_encoder,
            out_path=str(out_dir / f"confusion_matrix_{dataset_name}.png"),
            title=f"{model_name} — {dataset_name}",
        )

        record_results(
            run_id=run_id,
            model_name=model_name,
            train_on=train_on,
            test_on=dataset_name,
            metrics=metrics,
            train_info=train_info,
            config_path=config_path,
            notes=config.get("run_notes", ""),
        )
