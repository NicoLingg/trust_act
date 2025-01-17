import torch
import random
import numpy as np
from datetime import timedelta


def compute_dict_mean(epoch_dicts):
    """Compute the mean of a list of dictionaries."""
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_loss_string(losses: dict, prefix: str = "") -> str:
    """Format loss dictionary into a string, handling both tensor and non-tensor values."""
    parts = []
    if prefix:
        parts.append(f"{prefix}:")

    for k, v in losses.items():
        try:
            # Handle both tensor and non-tensor values
            if hasattr(v, "item"):
                value = v.item()
            else:
                value = float(v)
            value_str = f"{value:.4f}"
            parts.append(f"{k}={value_str}")
        except (TypeError, ValueError) as e:
            print(f"Warning: Could not format loss value for {k}: {v}")
            parts.append(f"{k}=NA")

    return " ".join(parts)


def format_time(seconds):
    """Convert seconds to a readable time format"""
    return str(timedelta(seconds=int(seconds)))
