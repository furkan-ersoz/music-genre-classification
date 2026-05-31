# ML Baseline — Handcrafted Features & Classical Models

This document covers the classical machine learning pipeline (`src/ml.py`) used as a baseline alongside the deep learning experiments.

## Feature Set

72 handcrafted features are extracted from 5-second non-overlapping audio segments using `librosa`. Features are computed per segment; train/test splits are performed at the **track level** to prevent data leakage.

| Group | Description | Count |
|---|---|---|
| MFCC | 13 coefficients, mean + variance | 26 |
| Chroma | 12 pitch classes, mean + variance | 24 |
| Spectral Contrast | 6 frequency bands, mean + variance | 12 |
| Spectral Centroid | Mean + variance | 2 |
| Spectral Bandwidth | Mean + variance | 2 |
| Spectral Rolloff | Mean + variance | 2 |
| Zero Crossing Rate | Mean + variance | 2 |
| RMS Energy | Mean + variance | 2 |
| **Total** | | **72** |

All features are standardized with `StandardScaler` fit on the training set before being passed to any model. This is particularly important for SVM and MLP, which are sensitive to feature scale.

## Models

| Model | Configuration |
|---|---|
| Random Forest | 200 estimators, unlimited depth, `n_jobs=-1` |
| SVM | RBF kernel, C=10, `gamma=scale`, probability estimates enabled |
| XGBoost | 200 estimators, max depth 6, learning rate 0.1 |
| MLP | Hidden layers (256, 128), early stopping, max 300 iterations |

XGBoost requires a separate install (`pip install xgboost`) and is skipped gracefully if unavailable.

## Running

```bash
python src/ml.py --config configs/mlTest.yaml
```

Config keys used by the ML pipeline:

```yaml
train_on: [gtzan, fma]
test_on:  [gtzan, fma]

gtzan_features: data/gtzan_features_manual.csv
fma_features:   data/fma_features_manual.csv

run_notes: ""
```

## Feature Extraction

Features are extracted separately from the deep learning pipeline:

```bash
python src/features.py --dataset both \
    --gtzan-root data/gtzan \
    --fma-audio   data/fma_small/fma_small/fma_small \
    --fma-metadata data/fma_small/fma_metadata/fma_metadata \
    --out-dir data
```

This produces `data/gtzan_features_manual.csv` and `data/fma_features_manual.csv`, each with columns `filename`, `track_id`, `label`, and the 72 feature columns.

## Results

Results are appended to `results/results_ml.csv` with columns: `run_id`, `model`, `train_on`, `test_on`, `accuracy`, `f1_macro`, `f1_weighted`, `precision_macro`, `recall_macro`, `train_time_sec`, `notes`.

Per-model confusion matrices are saved to `results/<run_id>/cm_<model>_<dataset>.png`.
