"""
Dynamic model loader.
Each model file must expose: get_model(num_classes, **kwargs) -> nn.Module
"""

import importlib
from torch import nn


def load_model(model_name: str, num_classes: int, **kwargs) -> nn.Module:
    """
    Dynamically import src/models/<model_name>.py and call get_model().

    Usage:
        model = load_model("model001", num_classes=10)
    """
    try:
        module = importlib.import_module(f"models.{model_name}")
    except ModuleNotFoundError:
        raise ValueError(f"Model '{model_name}' not found in src/models/")

    if not hasattr(module, "get_model"):
        raise AttributeError(
            f"models/{model_name}.py must define get_model(num_classes, **kwargs)"
        )

    return module.get_model(num_classes=num_classes, **kwargs)
