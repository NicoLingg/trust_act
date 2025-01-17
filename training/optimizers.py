import torch
from torch.optim import Optimizer
from typing import Tuple, Optional
from torch.optim.lr_scheduler import _LRScheduler
from training.schedulers import (
    build_lr_scheduler,
    build_kl_scheduler,
    KLLossWeightScheduler,
)


def build_optimizer_and_schedulers(
    policy, train_cfg, total_steps, steps_per_epoch
) -> Tuple[Optimizer, Optional[_LRScheduler], Optional[KLLossWeightScheduler]]:
    """Build optimizer with named parameter groups and scheduler."""

    if train_cfg.lr_backbone <= 0:
        print("Warning: Backbone learning rate <= 0, parameters will be frozen")

    backbone_params = [
        p for n, p in policy.named_parameters() if "backbone" in n and p.requires_grad
    ]
    transformer_params = [
        p
        for n, p in policy.named_parameters()
        if "backbone" not in n and p.requires_grad
    ]

    param_dicts = [
        {
            "params": transformer_params,
            "lr": train_cfg.learning_rate,
            "name": "transformer",  # Name for transformer parameters
        },
        {
            "params": backbone_params,
            "lr": train_cfg.lr_backbone,
            "name": "backbone",  # Name for backbone parameters
        },
    ]

    # Create optimizer
    optimizer_name = train_cfg.optimizer.lower()
    weight_decay = train_cfg.weight_decay

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(param_dicts, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(param_dicts, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    kl_scheduler = build_kl_scheduler(train_cfg, total_steps, steps_per_epoch)
    lr_scheduler = build_lr_scheduler(optimizer, train_cfg, total_steps)

    print(f"Using {optimizer_name.upper()} optimizer")
    print(f"• Transformer LR: {train_cfg.learning_rate:.2e}")
    print(f"• Backbone LR: {train_cfg.lr_backbone:.2e}")
    print(f"• Weight decay: {weight_decay:.2e}")
    if kl_scheduler is not None:
        print(f"• KL weight scheduler: {kl_scheduler.schedule_type}")
    else:
        print("• KL weight scheduler: None (constant weight)")

    return optimizer, lr_scheduler, kl_scheduler
