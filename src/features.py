"""
Feature extraction for music genre classification.
Extracts a unified feature set from audio files (WAV or MP3).

Output columns:
    filename, track_id, label + 61 features (MFCC x26, chroma x24, centroid x2,
    bandwidth x2, rolloff x2, zcr x2, rms x2, spectral_contrast x6, tempo removed)

Split strategy: always split on track_id, never on segments.
"""

import os
import warnings
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 22050
SEGMENT_DURATION = 5
N_MFCC = 13
N_CHROMA = 12
N_CONTRAST_BANDS = 6
HOP_LENGTH = 512
N_FFT = 2048


# ── Core extractor ────────────────────────────────────────────────────────────
def extract_features(audio: np.ndarray, sr: int) -> dict:
    """Extract feature dict from a single audio segment."""
    features = {}

    # MFCC (13 coefficients → mean + var = 26)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    for i in range(N_MFCC):
        features[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc{i+1}_var"]  = float(np.var(mfcc[i]))

    # Chroma (12 bins → mean + var = 24)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    for i in range(N_CHROMA):
        features[f"chroma{i+1}_mean"] = float(np.mean(chroma[i]))
        features[f"chroma{i+1}_var"]  = float(np.var(chroma[i]))

    # Spectral centroid (mean + var = 2)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr,
                                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_var"]  = float(np.var(centroid))

    # Spectral bandwidth (mean + var = 2)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr,
                                                    n_fft=N_FFT, hop_length=HOP_LENGTH)
    features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
    features["spectral_bandwidth_var"]  = float(np.var(bandwidth))

    # Spectral rolloff (mean + var = 2)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr,
                                                n_fft=N_FFT, hop_length=HOP_LENGTH)
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_var"]  = float(np.var(rolloff))

    # Zero crossing rate (mean + var = 2)
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_var"]  = float(np.var(zcr))

    # RMS energy (mean + var = 2)
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_var"]  = float(np.var(rms))

    # Spectral contrast (6 bands → mean + var = 12)
    # More stable than tempo for 5-second segments
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr,
                                                  n_fft=N_FFT, hop_length=HOP_LENGTH,
                                                  n_bands=N_CONTRAST_BANDS)
    for i in range(N_CONTRAST_BANDS):
        features[f"spectral_contrast{i+1}_mean"] = float(np.mean(contrast[i]))
        features[f"spectral_contrast{i+1}_var"]  = float(np.var(contrast[i]))

    return features  # 26 + 24 + 2 + 2 + 2 + 2 + 2 + 12 = 72 features


def slice_audio(audio: np.ndarray, sr: int,
                duration: int = SEGMENT_DURATION) -> list[np.ndarray]:
    """Split audio into fixed-length non-overlapping segments."""
    seg_len = sr * duration
    return [audio[start: start + seg_len]
            for start in range(0, len(audio) - seg_len + 1, seg_len)]


# ── GTZAN ─────────────────────────────────────────────────────────────────────
def extract_gtzan(gtzan_root: str, output_path: str) -> None:
    """
    Expects structure:
        gtzan_root/genres_original/<genre>/<track>.wav

    Split note: always split on track_id to avoid data leakage.
    """
    genres_dir = Path(gtzan_root) / "genres_original"
    rows = []

    genres = sorted([d.name for d in genres_dir.iterdir() if d.is_dir()])
    print(f"GTZAN genres ({len(genres)}): {genres}")

    for genre in genres:
        files = sorted([f for f in (genres_dir / genre).glob("*.wav") 
                if not f.name.startswith("._")])
        for fpath in tqdm(files, desc=f"GTZAN {genre}"):
            try:
                audio, sr = librosa.load(str(fpath), sr=SAMPLE_RATE, mono=True)
            except Exception as e:
                print(f"  skip {fpath.name}: {e}")
                continue

            track_id = fpath.stem  # e.g. "blues00000"
            for seg_idx, segment in enumerate(slice_audio(audio, sr)):
                feats = extract_features(segment, sr)
                feats["filename"] = f"{track_id}_seg{seg_idx}"
                feats["track_id"] = track_id
                feats["label"]    = genre
                rows.append(feats)

    df = pd.DataFrame(rows)
    cols = ["filename", "track_id", "label"] + [
        c for c in df.columns if c not in ("filename", "track_id", "label")
    ]
    df[cols].to_csv(output_path, index=False)
    print(f"\nGTZAN → {output_path}  ({len(df)} rows, {len(df.columns)} cols)")
    print(f"Unique tracks: {df['track_id'].nunique()}")


# ── FMA-Small ─────────────────────────────────────────────────────────────────
def extract_fma(fma_audio_root: str, fma_metadata_root: str,
                output_path: str) -> None:
    """
    Expects:
        fma_audio_root/    → 000/ 001/ ... (mp3 files)
        fma_metadata_root/ → tracks.csv

    Split note: always split on track_id to avoid data leakage.
    """
    tracks_csv = Path(fma_metadata_root) / "tracks.csv"
    tracks     = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    genre_map  = tracks["track"]["genre_top"].dropna()

    audio_root = Path(fma_audio_root)
    mp3_files  = {}
    for folder in audio_root.iterdir():
        if not folder.is_dir():
            continue
        for f in folder.iterdir():
            if f.suffix == ".mp3" and not f.name.startswith("._"):
                try:
                    mp3_files[int(f.stem)] = f
                except ValueError:
                    pass

    valid_ids = [tid for tid in mp3_files if tid in genre_map.index]
    print(f"FMA-Small: {len(valid_ids)} tracks with genre labels")

    rows = []
    for tid in tqdm(valid_ids, desc="FMA-Small"):
        fpath = mp3_files[tid]
        label = genre_map[tid]
        try:
            audio, sr = librosa.load(str(fpath), sr=SAMPLE_RATE, mono=True, duration=30)
        except Exception as e:
            print(f"  skip {fpath.name}: {e}")
            continue

        track_id = str(tid)
        for seg_idx, segment in enumerate(slice_audio(audio, sr)):
            feats = extract_features(segment, sr)
            feats["filename"] = f"{track_id}_seg{seg_idx}"
            feats["track_id"] = track_id
            feats["label"]    = label
            rows.append(feats)

    df = pd.DataFrame(rows)
    cols = ["filename", "track_id", "label"] + [
        c for c in df.columns if c not in ("filename", "track_id", "label")
    ]
    df[cols].to_csv(output_path, index=False)
    print(f"\nFMA-Small → {output_path}  ({len(df)} rows, {len(df.columns)} cols)")
    print(f"Unique tracks: {df['track_id'].nunique()}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features from audio datasets")
    parser.add_argument("--dataset",      choices=["gtzan", "fma", "both"], default="both")
    parser.add_argument("--gtzan-root",   default="data/gtzan")
    parser.add_argument("--fma-audio",    default="data/fma_small/fma_small/fma_small")
    parser.add_argument("--fma-metadata", default="data/fma_small/fma_metadata/fma_metadata")
    parser.add_argument("--out-dir",      default="data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset in ("gtzan", "both"):
        extract_gtzan(
            gtzan_root=args.gtzan_root,
            output_path=os.path.join(args.out_dir, "gtzan_features_manual.csv")
        )

    if args.dataset in ("fma", "both"):
        extract_fma(
            fma_audio_root=args.fma_audio,
            fma_metadata_root=args.fma_metadata,
            output_path=os.path.join(args.out_dir, "fma_features_manual.csv")
        )