# Music Genre Classification

Music genre classification using PyTorch — comparing hybrid deep learning architectures and classical ML baselines on GTZAN and FMA-Small datasets.

## Project Structure

```
├── literature/             # Papers and notes (gitignored)
├── data/                   # Datasets (gitignored)
├── src/
│   ├── dataset.py          # DataLoader factory, label normalization, track-level splits
│   ├── spectrogram.py      # Log-mel spectrogram generation (224×224 RGB PNG)
│   ├── features.py         # Handcrafted feature extraction (72 features, librosa)
│   ├── models/
│   │   ├── __init__.py     # Dynamic model loader
│   │   ├── model001.py     # EfficientNet-B0 + Temporal Attention
│   │   └── model002.py     # ResNet18 + Bidirectional GRU + Attention Pooling
│   ├── train.py            # Training loop (early stopping, LR scheduling, checkpointing)
│   ├── evaluate.py         # Metrics, confusion matrices, training curves
│   └── ml.py               # Classical ML pipeline (RF, SVM, XGBoost, MLP)
├── configs/                # Experiment configs (YAML)
├── results/                # Per-run metrics, plots, confusion matrices, checkpoints
├── reports/                # Assignment reports
└── run_all.py              # Batch experiment runner
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Deep learning experiments

```bash
python run_all.py --config configs/defaultTest.yaml
```

### Classical ML experiments

```bash
python src/ml.py --config configs/mlTest.yaml
```

### Preprocessing (run once)

```bash
# Generate log-mel spectrograms
python src/spectrogram.py --dataset both

# Extract handcrafted features
python src/features.py --dataset both
```

## Config Format

Experiments are fully controlled by YAML config files. Example:

```yaml
models: [model001, model002]
train_on: [gtzan, fma]
test_on:  [gtzan, fma]

gtzan_labels:  data/GTZAN/spectrogramsManuel/labels.csv
gtzan_img_dir: data/GTZAN/spectrogramsManuel/
fma_labels:    data/FMA_Small/spectrogramsManuel/labels.csv
fma_img_dir:   data/FMA_Small/spectrogramsManuel/

epochs: 50
lr: 1e-3
batch_size: 32
early_stopping_patience: 10
scheduler: plateau        # plateau | cosine
img_size: 224
num_workers: 2
run_notes: ""
```

Multiple models in a single config are run sequentially; each gets its own `run_id` and results directory.

## Deep Learning Models

Both models take 224×224 log-mel spectrogram images as input and output per-class logits.

### model001 — EfficientNet-B0 + Temporal Attention

EfficientNet-B0 pretrained on ImageNet extracts a `(B, 1280, 7, 7)` feature map. The width dimension is treated as a time sequence, projected to 256 dimensions, and passed through two layers of multi-head self-attention (`num_heads=4`). A mean pool over the time axis feeds the classifier head.

The self-attention mechanism lets the model weigh which time slices of the spectrogram matter most for the genre decision — without imposing a fixed sequential order.

### model002 — ResNet18 + Bidirectional GRU + Attention Pooling

ResNet18 (avgpool and fc removed) extracts spatial features. The frequency dimension is collapsed via adaptive average pooling, leaving a sequence of time-step vectors `(B, W', 512)`. A two-layer Bidirectional GRU processes this sequence in both directions, capturing temporal context from past and future simultaneously. An attention pooling layer then computes a learned weighted sum over time steps before the classifier head.

The bidirectional pass means each time step is contextualized by the entire sequence — useful for genres where rhythmic structure or phase transitions are diagnostic.

| | model001 | model002 |
|---|---|---|
| Backbone | EfficientNet-B0 | ResNet18 |
| Temporal module | Multi-head self-attention | Bidirectional GRU |
| Aggregation | Mean pool | Attention pooling |
| Parameter count | Higher | Lower |
| Strength | Global tonal patterns | Rhythmic / sequential structure |

## Data Pipeline

### Spectrograms

Log-mel spectrograms are computed from 5-second non-overlapping audio segments using `librosa` (n_mels=128, n_fft=2048, hop_length=512) and saved as 224×224 RGB PNG files compatible with torchvision pretrained models.

Training images use random horizontal flip and color jitter for augmentation. Validation and test images use only resize and normalize.

### Label Normalization

GTZAN and FMA-Small use different label conventions. Labels are normalized to a shared title-case vocabulary (e.g. `hiphop` → `Hip-Hop`, `rock` → `Rock`). FMA categories `International` and `Instrumental` are excluded. The label encoder is fit on training labels only; test segments with unseen labels are dropped with a warning.

### Split Strategy

All splits are performed at the **track level** to prevent data leakage — segments from the same track never appear in both train and test sets.

| Split | Size |
|---|---|
| Train | 80% of tracks |
| Validation | 10% of tracks |
| Test | 20% of tracks (evaluated per dataset) |

## Training

The training loop (`train.py`) supports:

- **Early stopping** — configurable patience (default 10 epochs)
- **LR scheduling** — `ReduceLROnPlateau` (default) or `CosineAnnealingLR`
- **Checkpointing** — best validation accuracy checkpoint saved per run
- **History logging** — per-epoch train/val loss and accuracy saved to `history.json`

## Evaluation & Results

Each run produces a dedicated directory under `results/<run_id>/` containing:

- `best_model.pt` — best checkpoint by validation accuracy
- `training_curves.png` — loss and accuracy over epochs
- `confusion_matrix_<dataset>.png` — per dataset
- One row appended to `results/results.csv` per (model, test dataset) combination

`results/results.csv` columns: `run_id`, `model`, `train_on`, `test_on`, `epoch`, `accuracy`, `f1_macro`, `f1_weighted`, `precision_macro`, `recall_macro`, `loss`, `train_time_sec`, `best_val_accuracy`, `config_path`, `notes`.

The ML pipeline writes analogous results to `results/results_ml.csv`.

## Classical ML Baseline

See [`MLExplanations.md`](MLExplanations.md) for details on the handcrafted feature set and classical model configurations.
