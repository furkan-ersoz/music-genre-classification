# Music Genre Classification

Music genre classification using PyTorch — comparing CNN baseline, BiLSTM, and CNN+Transformer hybrid architectures on GTZAN and FMA-Small datasets.

## Project Structure
├── literature/         # Papers and notes (gitignored)
├── data/               # Datasets (gitignored)
├── src/
│   ├── dataset.py      # DataLoader factory
│   ├── spectrogram.py  # Mel-spec, STFT, MFCC converters
│   ├── features.py     # Feature extraction for ML
│   ├── models/
│   │   ├── model001.py # CNN baseline (transfer learning)
│   │   ├── model002.py # BiLSTM
│   │   └── model003.py # CNN + Transformer hybrid
│   ├── train.py        # Training loop
│   └── evaluate.py     # Metrics and reporting
├── configs/            # Experiment configs (YAML)
├── results/            # Metrics, plots, confusion matrices
├── reports/            # Assignment reports
└── run_all.py          # Batch experiment runner

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_all.py --config configs/test.yaml
```

## Feature Set

72 handcrafted features extracted from 5-second non-overlapping audio segments using `librosa`. Features are computed per segment; train/test splits are performed at the **track level** to prevent data leakage.

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

## ML Models

Four classical machine learning models are trained and compared:

| Model | Notes |
|---|---|
| Random Forest | Ensemble of decision trees |
| SVM | RBF kernel, probability estimates enabled |
| XGBoost | Gradient boosted trees |
| MLP | Shallow neural network (sklearn) |

## Split Strategy

- **Train:** 80% of GTZAN + 80% of FMA-Small (combined)
- **Test:** 20% of GTZAN and 20% of FMA-Small (evaluated separately)
- **Validation:** 10% held out from train set for hyperparameter tuning
- Splits are performed at the **track level**, not segment level, to prevent leakage
