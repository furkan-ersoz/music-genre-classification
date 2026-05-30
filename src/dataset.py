"""
Dataset and DataLoader factory for music genre classification.

Label strategy:
    - GTZAN labels normalized to title-case to match FMA
    - FMA: International and Instrumental excluded
    - Shared labels: Hip-Hop, Pop, Rock (normalized from both sides)
    - Union of all remaining labels used when training on both datasets
    - Split is always performed at track level to prevent data leakage
"""

import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── Label config ──────────────────────────────────────────────────────────────
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


# ── Transform pipelines ───────────────────────────────────────────────────────
def get_transforms(img_size: int = 224, augment: bool = False):
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── Dataset classes (module-level for multiprocessing pickle) ─────────────────
class SpectrogramDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str,
                 label_encoder: LabelEncoder, transform=None):
        self.df            = df.reset_index(drop=True)
        self.img_dir       = Path(img_dir)
        self.label_encoder = label_encoder
        self.transform     = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = self.img_dir / row["spec_filename"]
        image    = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.label_encoder.transform([row["label"]])[0]
        return image, label


class ConcatSpectrogramDataset(Dataset):
    """Concatenates multiple SpectrogramDatasets — defined at module level
    so multiprocessing workers can pickle it."""
    def __init__(self, datasets: list):
        self.datasets = datasets
        self.lengths  = [len(d) for d in datasets]

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        for ds, length in zip(self.datasets, self.lengths):
            if idx < length:
                return ds[idx]
            idx -= length


# ── Label loading helpers ─────────────────────────────────────────────────────
def load_gtzan_labels(labels_csv: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    df["label"]   = df["label"].map(LABEL_MAPPING).fillna(df["label"])
    df["dataset"] = "gtzan"
    return df


def load_fma_labels(labels_csv: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    df = df[~df["label"].isin(FMA_EXCLUDE)].copy()
    df["label"]   = df["label"].map(LABEL_MAPPING).fillna(df["label"])
    df["dataset"] = "fma"
    return df


# ── Track-level split ─────────────────────────────────────────────────────────
def track_level_split(df: pd.DataFrame,
                      test_size: float = 0.2,
                      val_size: float = 0.1,
                      random_state: int = 42):
    track_ids = df["song_id"].unique()
    train_val_ids, test_ids = train_test_split(
        track_ids, test_size=test_size, random_state=random_state)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_size / (1 - test_size),
        random_state=random_state)

    return (df[df["song_id"].isin(train_ids)].copy(),
            df[df["song_id"].isin(val_ids)].copy(),
            df[df["song_id"].isin(test_ids)].copy())


# ── Factory ───────────────────────────────────────────────────────────────────
def build_dataloaders(config: dict):
    img_size    = config.get("img_size", 224)
    batch_size  = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 2)

    # MPS doesn't support pin_memory
    pin_memory = torch.cuda.is_available()

    dataset_loaders = {
        "gtzan": lambda: load_gtzan_labels(config["gtzan_labels"]),
        "fma":   lambda: load_fma_labels(config["fma_labels"]),
    }
    dataset_dirs = {
        "gtzan": config["gtzan_img_dir"],
        "fma":   config["fma_img_dir"],
    }

    # Split each dataset at track level
    all_names = set(config["train_on"]) | set(config["test_on"])
    splits    = {}
    for name in all_names:
        df = dataset_loaders[name]()
        splits[name] = track_level_split(df)

    train_parts, val_parts = [], []
    for name in config["train_on"]:
        train_df, val_df, _ = splits[name]
        train_parts.append((train_df, dataset_dirs[name]))
        val_parts.append((val_df, dataset_dirs[name]))

    test_dfs = {}
    for name in config["test_on"]:
        _, _, test_df = splits[name]
        test_dfs[name] = (test_df, dataset_dirs[name])

    # Fit label encoder on all training labels
    all_train_labels = pd.concat([df for df, _ in train_parts])["label"]
    le = LabelEncoder()
    le.fit(sorted(all_train_labels.unique()))
    num_classes = len(le.classes_)

    train_transform = get_transforms(img_size, augment=True)
    val_transform   = get_transforms(img_size, augment=False)

    train_dataset = ConcatSpectrogramDataset([
        SpectrogramDataset(df, img_dir, le, train_transform)
        for df, img_dir in train_parts
    ])
    val_dataset = ConcatSpectrogramDataset([
        SpectrogramDataset(df, img_dir, le, val_transform)
        for df, img_dir in val_parts
    ])

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=pin_memory)

    # Test loaders — per dataset
    test_loaders = {}
    for name, (test_df, img_dir) in test_dfs.items():
        known   = set(le.classes_)
        before  = len(test_df)
        test_df = test_df[test_df["label"].isin(known)].copy()
        after   = len(test_df)
        if before != after:
            print(f"  [{name}] dropped {before-after} segments with unseen labels")

        ds = SpectrogramDataset(test_df, img_dir, le, val_transform)
        test_loaders[name] = DataLoader(ds, batch_size=batch_size,
                                        shuffle=False, num_workers=num_workers,
                                        pin_memory=pin_memory)

    return train_loader, val_loader, test_loaders, le, num_classes