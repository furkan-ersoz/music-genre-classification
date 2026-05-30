"""
Log-Mel spectrogram generation for music genre classification.
Produces 224x224 RGB PNG images from 5-second audio segments.

Output structure:
    data/GTZAN/spectrogramsManuel/
        <track_id>_seg<n>.png
        labels.csv  → song_id, segment_id, spec_filename, label
    data/FMA_Small/spectrogramsManuel/
        <track_id>_seg<n>.png
        labels.csv  → song_id, segment_id, spec_filename, label

Spectrogram type : Log-Mel (power_to_db applied)
Parameters       : n_mels=128, n_fft=2048, hop_length=512
Output size      : 224x224 RGB PNG (ResNet/EfficientNet compatible)
Segment duration : 5 seconds, non-overlapping
Split strategy   : always split on track_id (not segment) to prevent leakage
"""

import os
import warnings
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 22050
SEGMENT_DURATION = 5
N_MELS           = 128
N_FFT            = 2048
HOP_LENGTH       = 512
IMG_SIZE         = 224          # px, square


# ── Core helpers ──────────────────────────────────────────────────────────────
def slice_audio(audio: np.ndarray, sr: int,
                duration: int = SEGMENT_DURATION) -> list[np.ndarray]:
    """Split audio into fixed-length non-overlapping segments."""
    seg_len = sr * duration
    return [audio[start: start + seg_len]
            for start in range(0, len(audio) - seg_len + 1, seg_len)]


def save_logmel_png(segment: np.ndarray, sr: int, out_path: str) -> None:
    """Compute log-mel spectrogram and save as 224x224 RGB PNG."""
    mel = librosa.feature.melspectrogram(
        y=segment, sr=sr,
        n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(IMG_SIZE / 100, IMG_SIZE / 100), dpi=100)
    librosa.display.specshow(log_mel, sr=sr, hop_length=HOP_LENGTH,
                             x_axis=None, y_axis=None, ax=ax, cmap="viridis")
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, format="png", bbox_inches="tight",
                pad_inches=0, dpi=100)
    plt.close(fig)


# ── GTZAN ─────────────────────────────────────────────────────────────────────
def generate_gtzan(gtzan_root: str, out_root: str) -> None:
    """
    Expects:
        gtzan_root/genres_original/<genre>/<track>.wav
    Produces:
        out_root/GTZAN/spectrogramsManuel/<track_id>_seg<n>.png
        out_root/GTZAN/spectrogramsManuel/labels.csv
    """
    genres_dir = Path(gtzan_root) / "genres_original"
    out_dir    = Path(out_root) / "GTZAN" / "spectrogramsManuel"
    out_dir.mkdir(parents=True, exist_ok=True)

    genres = sorted([d.name for d in genres_dir.iterdir() if d.is_dir()])
    print(f"GTZAN genres ({len(genres)}): {genres}")

    rows = []
    for genre in genres:
        files = sorted([f for f in (genres_dir / genre).glob("*.wav")
                        if not f.name.startswith("._")])
        for fpath in tqdm(files, desc=f"GTZAN {genre}"):
            try:
                audio, sr = librosa.load(str(fpath), sr=SAMPLE_RATE, mono=True)
            except Exception as e:
                print(f"  skip {fpath.name}: {e}")
                continue

            track_id = fpath.stem
            for seg_idx, segment in enumerate(slice_audio(audio, sr)):
                spec_filename = f"{track_id}_seg{seg_idx}.png"
                out_path      = out_dir / spec_filename
                try:
                    save_logmel_png(segment, sr, str(out_path))
                except Exception as e:
                    print(f"  spec error {spec_filename}: {e}")
                    continue
                rows.append({
                    "song_id":       track_id,
                    "segment_id":    seg_idx,
                    "spec_filename": spec_filename,
                    "label":         genre,
                })

    labels_df = pd.DataFrame(rows, columns=["song_id", "segment_id",
                                             "spec_filename", "label"])
    labels_df.to_csv(out_dir / "labels.csv", index=False)
    print(f"\nGTZAN → {out_dir}")
    print(f"  {len(labels_df)} spectrograms, {labels_df['song_id'].nunique()} tracks")


# ── FMA-Small ─────────────────────────────────────────────────────────────────
def generate_fma(fma_audio_root: str, fma_metadata_root: str,
                 out_root: str) -> None:
    """
    Expects:
        fma_audio_root/    → 000/ 001/ ... (mp3 files)
        fma_metadata_root/ → tracks.csv
    Produces:
        out_root/FMA_Small/spectrogramsManuel/<track_id>_seg<n>.png
        out_root/FMA_Small/spectrogramsManuel/labels.csv
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

    out_dir = Path(out_root) / "FMA_Small" / "spectrogramsManuel"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for tid in tqdm(valid_ids, desc="FMA-Small"):
        fpath = mp3_files[tid]
        label = genre_map[tid]
        try:
            audio, sr = librosa.load(str(fpath), sr=SAMPLE_RATE,
                                     mono=True, duration=30)
        except Exception as e:
            print(f"  skip {fpath.name}: {e}")
            continue

        track_id = str(tid)
        for seg_idx, segment in enumerate(slice_audio(audio, sr)):
            spec_filename = f"{track_id}_seg{seg_idx}.png"
            out_path      = out_dir / spec_filename
            try:
                save_logmel_png(segment, sr, str(out_path))
            except Exception as e:
                print(f"  spec error {spec_filename}: {e}")
                continue
            rows.append({
                "song_id":       track_id,
                "segment_id":    seg_idx,
                "spec_filename": spec_filename,
                "label":         label,
            })

    labels_df = pd.DataFrame(rows, columns=["song_id", "segment_id",
                                             "spec_filename", "label"])
    labels_df.to_csv(out_dir / "labels.csv", index=False)
    print(f"\nFMA-Small → {out_dir}")
    print(f"  {len(labels_df)} spectrograms, {labels_df['song_id'].nunique()} tracks")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate log-mel spectrograms from audio datasets")
    parser.add_argument("--dataset",      choices=["gtzan", "fma", "both"],
                        default="both")
    parser.add_argument("--gtzan-root",   default="data/gtzan")
    parser.add_argument("--fma-audio",    default="data/fma_small/fma_small/fma_small")
    parser.add_argument("--fma-metadata", default="data/fma_small/fma_metadata/fma_metadata")
    parser.add_argument("--out-root",     default="data")
    args = parser.parse_args()

    if args.dataset in ("gtzan", "both"):
        generate_gtzan(
            gtzan_root=args.gtzan_root,
            out_root=args.out_root,
        )

    if args.dataset in ("fma", "both"):
        generate_fma(
            fma_audio_root=args.fma_audio,
            fma_metadata_root=args.fma_metadata,
            out_root=args.out_root,
        )