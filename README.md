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
