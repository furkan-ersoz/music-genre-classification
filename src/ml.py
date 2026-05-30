"""
ML pipeline for music genre classification.
Uses handcrafted features (72 features per 5-second segment).

Models:
    - Random Forest
    - SVM (RBF kernel)
    - XGBoost
    - MLP (sklearn)

Split strategy : track-level (no data leakage)
Train          : concat of selected datasets (80%)
Validation     : 10% (for hyperparameter tuning)
Test           : per-dataset 20% evaluated separately

Results appended to results/results_ml.csv
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix)
import csv

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not installed, XGBoost will be skipped.")

# ── Label config (same as dataset.py) ────────────────────────────────────────
LABEL_MAPPING = {
    "blues":        "Blues",
    "classical":    "Classical",
    "country":      "Country",
    "disco":        "Disco",
    "hiphop":       "Hip-Hop",
    "jazz":         "Jazz",
    "metal":        "Metal",
    "pop":          "Pop",
    "reggae":       "Reggae",
    "rock":         "Rock",
    "Hip-Hop":      "Hip-Hop",
    "Pop":          "Pop",
    "Rock":         "Rock",
    "Electronic":   "Electronic",
    "Experimental": "Experimental",
    "Folk":         "Folk",
}

FMA_EXCLUDE = {"International", "Instrumental"}

FEATURE_COLS = [
    *[f"mfcc{i}_{s}"         for i in range(1, 14) for s in ("mean", "var")],
    *[f"chroma{i}_{s}"       for i in range(1, 13) for s in ("mean", "var")],
    "spectral_centroid_mean", "spectral_centroid_var",
    "spectral_bandwidth_mean","spectral_bandwidth_var",
    "spectral_rolloff_mean",  "spectral_rolloff_var",
    "zcr_mean",               "zcr_var",
    "rms_mean",               "rms_var",
    *[f"spectral_contrast{i}_{s}" for i in range(1, 7) for s in ("mean", "var")],
]

RESULTS_CSV = Path("results") / "results_ml.csv"
CSV_COLUMNS = [
    "run_id", "model", "train_on", "test_on",
    "accuracy", "f1_macro", "f1_weighted",
    "precision_macro", "recall_macro",
    "train_time_sec", "notes",
]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_gtzan(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["label"]   = df["label"].map(LABEL_MAPPING).fillna(df["label"])
    df["dataset"] = "gtzan"
    return df


def load_fma(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[~df["label"].isin(FMA_EXCLUDE)].copy()
    df["label"]   = df["label"].map(LABEL_MAPPING).fillna(df["label"])
    df["dataset"] = "fma"
    return df


# ── Track-level split ─────────────────────────────────────────────────────────
def track_level_split(df: pd.DataFrame,
                      test_size: float = 0.2,
                      val_size: float = 0.1,
                      random_state: int = 42):
    track_ids = df["track_id"].unique()
    train_val_ids, test_ids = train_test_split(
        track_ids, test_size=test_size, random_state=random_state)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_size / (1 - test_size),
        random_state=random_state)
    return (df[df["track_id"].isin(train_ids)].copy(),
            df[df["track_id"].isin(val_ids)].copy(),
            df[df["track_id"].isin(test_ids)].copy())


# ── Model definitions ─────────────────────────────────────────────────────────
def get_models():
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=None,
            n_jobs=-1, random_state=42),
        "svm": SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, random_state=42),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=300, random_state=42,
            early_stopping=True, validation_fraction=0.1),
    }
    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=200, max_depth=6,
            learning_rate=0.1, use_label_encoder=False,
            eval_metric="mlogloss", random_state=42,
            n_jobs=-1)
    return models


# ── Results helpers ───────────────────────────────────────────────────────────
def ensure_csv_header():
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_CSV.exists():
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def record_result(run_id, model_name, train_on, test_on,
                  metrics, train_time, notes=""):
    ensure_csv_header()
    row = {
        "run_id":          run_id,
        "model":           model_name,
        "train_on":        "+".join(train_on),
        "test_on":         test_on,
        "accuracy":        metrics["accuracy"],
        "f1_macro":        metrics["f1_macro"],
        "f1_weighted":     metrics["f1_weighted"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro":    metrics["recall_macro"],
        "train_time_sec":  round(train_time, 1),
        "notes":           notes,
    }
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)
    print(f"  Result recorded → {RESULTS_CSV}")


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy":        round(accuracy_score(y_true, y_pred), 4),
        "f1_macro":        round(f1_score(y_true, y_pred, average="macro",    zero_division=0), 4),
        "f1_weighted":     round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro":    round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }


def save_confusion_matrix(y_true, y_pred, class_names,
                          out_path: str, title: str = "") -> None:
    cm = confusion_matrix(y_true, y_pred)
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


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_ml(config: dict, config_path: str) -> None:
    train_on   = config.get("train_on", [])
    test_on    = config.get("test_on",  [])
    notes      = config.get("run_notes", "")
    gtzan_csv  = config.get("gtzan_features", "data/gtzan_features_manual.csv")
    fma_csv    = config.get("fma_features",   "data/fma_features_manual.csv")

    loaders = {"gtzan": lambda: load_gtzan(gtzan_csv),
               "fma":   lambda: load_fma(fma_csv)}

    print(f"\n{'='*60}")
    print(f"Train on : {train_on}")
    print(f"Test on  : {test_on}")
    print(f"{'='*60}\n")

    # Split each dataset at track level
    all_names = set(train_on) | set(test_on)
    splits = {}
    for name in all_names:
        df = loaders[name]()
        # Only keep feature columns that exist in the CSV
        available = [c for c in FEATURE_COLS if c in df.columns]
        splits[name] = (track_level_split(df), df, available)

    # Concat train parts
    train_dfs = []
    for name in train_on:
        (train_df, val_df, _), df, feats = splits[name]
        train_dfs.append(train_df)

    train_all = pd.concat(train_dfs, ignore_index=True)

    # Feature columns — intersection across all datasets
    feat_cols = [c for c in FEATURE_COLS if c in train_all.columns]
    print(f"Feature columns: {len(feat_cols)}")

    # Label encoder
    le = LabelEncoder()
    le.fit(sorted(train_all["label"].unique()))
    print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")

    X_train = train_all[feat_cols].values
    y_train = le.transform(train_all["label"].values)

    # Scale (important for SVM and MLP)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Test sets per dataset
    test_sets = {}
    for name in test_on:
        (_, _, test_df), df, feats = splits[name]
        known   = set(le.classes_)
        test_df = test_df[test_df["label"].isin(known)].copy()
        X_test  = scaler.transform(test_df[feat_cols].values)
        y_test  = le.transform(test_df["label"].values)
        test_sets[name] = (X_test, y_test)
        print(f"Test [{name}]: {len(test_df)} segments")

    # Run each model
    models  = get_models()
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S") + "_ml"
    out_dir = Path("results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name, clf in models.items():
        print(f"\n{'─'*60}")
        print(f"Training: {model_name}")

        import time
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f"  Train time: {train_time:.1f}s")

        for dataset_name, (X_test, y_test) in test_sets.items():
            y_pred  = clf.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)

            print(f"  [{dataset_name}] accuracy={metrics['accuracy']}  "
                  f"f1_macro={metrics['f1_macro']}")

            save_confusion_matrix(
                y_test, y_pred,
                class_names=le.classes_,
                out_path=str(out_dir / f"cm_{model_name}_{dataset_name}.png"),
                title=f"{model_name} — {dataset_name}",
            )

            record_result(
                run_id=run_id,
                model_name=model_name,
                train_on=train_on,
                test_on=dataset_name,
                metrics=metrics,
                train_time=train_time,
                notes=notes,
            )

    print(f"\nDone. Results → {RESULTS_CSV}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML pipeline for MGC")
    parser.add_argument("--config", default="configs/mlTest.yaml")
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Config not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_ml(config, args.config)