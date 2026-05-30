"""
Batch experiment runner for music genre classification.

Usage:
    python run_all.py                            # uses configs/defaultTest.yaml
    python run_all.py --config configs/custom.yaml
"""

import argparse
import sys
import torch
import yaml
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import build_dataloaders
from models  import load_model
from train   import train_model
from evaluate import run_evaluation


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_experiment(config: dict, config_path: str, device: torch.device) -> None:
    models_list = config.get("models", [])
    train_on    = config.get("train_on", [])
    test_on     = config.get("test_on",  [])

    if not models_list:
        raise ValueError("Config must specify at least one model under 'models:'")
    if not train_on:
        raise ValueError("Config must specify 'train_on' list")
    if not test_on:
        raise ValueError("Config must specify 'test_on' list")

    print(f"\n{'='*60}")
    print(f"Models   : {models_list}")
    print(f"Train on : {train_on}")
    print(f"Test on  : {test_on}")
    print(f"Device   : {device}")
    print(f"{'='*60}\n")

    print("Loading datasets...")
    train_loader, val_loader, test_loaders, label_encoder, num_classes = \
        build_dataloaders(config)

    print(f"Classes ({num_classes}): {list(label_encoder.classes_)}")
    print(f"Train segments : {len(train_loader.dataset)}")
    print(f"Val segments   : {len(val_loader.dataset)}")
    for name, loader in test_loaders.items():
        print(f"Test [{name}]    : {len(loader.dataset)} segments")

    for model_name in models_list:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{model_name}"
        print(f"\n{'─'*60}")
        print(f"Model : {model_name}  |  run_id : {run_id}")
        print(f"{'─'*60}")

        model = load_model(model_name, num_classes=num_classes)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        train_info = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            run_id=run_id,
            device=device,
        )

        model.load_state_dict(
            torch.load(train_info["checkpoint_path"], map_location=device, weights_only=True)
        )

        run_evaluation(
            model=model,
            test_loaders=test_loaders,
            label_encoder=label_encoder,
            train_info=train_info,
            run_id=run_id,
            model_name=model_name,
            train_on=train_on,
            config_path=config_path,
            config=config,
            device=device,
        )


def main():
    parser = argparse.ArgumentParser(description="MGC batch experiment runner")
    parser.add_argument("--config", default="configs/defaultTest.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    config_path = args.config
    if not Path(config_path).exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = get_device()
    run_experiment(config, config_path, device)


if __name__ == "__main__":
    main()
