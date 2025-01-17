import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any, List, Optional


class MultiGroupWarmupScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        decay_strategy: str = "cosine",
        min_lr_ratios: Dict[str, float] = None,
        warmup_start_lr_ratios: Dict[str, float] = None,
        step_size: int = None,
        step_gamma: float = None,
        last_step: int = -1,
    ):
        # Store step-based parameters before parent init
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_strategy = decay_strategy
        self.step_size = step_size
        self.step_gamma = step_gamma

        # Initialize learning rate ratios
        default_min_ratio = 0.1
        default_warmup_ratio = 0.01

        self.min_lr_ratios = min_lr_ratios or {
            "backbone": default_min_ratio,
            "transformer": default_min_ratio,
        }
        self.warmup_start_lr_ratios = warmup_start_lr_ratios or {
            "backbone": default_warmup_ratio,
            "transformer": default_warmup_ratio,
        }

        # Validate parameters
        if warmup_steps >= total_steps:
            raise ValueError("warmup_steps must be less than total_steps")
        if decay_strategy not in ["cosine", "step", "none"]:
            raise ValueError(f"Unknown decay strategy: {decay_strategy}")
        if decay_strategy == "step" and (step_size is None or step_gamma is None):
            raise ValueError("step_size and step_gamma required for step decay")

        # Store initial learning rates and validate group names
        self.base_lrs_dict = {}
        self.group_names = []
        for i, group in enumerate(optimizer.param_groups):
            name = group.get("name", f"group_{i}")
            self.group_names.append(name)
            self.base_lrs_dict[name] = group["lr"]

            if name not in self.min_lr_ratios:
                raise ValueError(f"No min_lr_ratio specified for group {name}")
            if name not in self.warmup_start_lr_ratios:
                raise ValueError(f"No warmup_start_ratio specified for group {name}")

        super().__init__(optimizer, last_epoch=last_step)

    def get_lr(self) -> List[float]:
        """Calculate learning rates based on current step."""
        if not len(self.optimizer.param_groups) == len(self.group_names):
            raise ValueError("Number of parameter groups changed!")

        lrs = []
        current_step = self.last_epoch  # This is actually storing steps

        for group_name in self.group_names:
            base_lr = self.base_lrs_dict[group_name]
            min_lr = self.min_lr_ratios[group_name] * base_lr
            warmup_start_lr = self.warmup_start_lr_ratios[group_name] * base_lr

            if current_step < self.warmup_steps:
                # Linear warmup
                progress = float(current_step) / float(max(1, self.warmup_steps))
                lr = warmup_start_lr + progress * (base_lr - warmup_start_lr)
            else:
                if self.decay_strategy == "none":
                    lr = base_lr
                elif self.decay_strategy == "cosine":
                    # Cosine decay from base_lr to min_lr
                    progress = float(current_step - self.warmup_steps) / float(
                        max(1, self.total_steps - self.warmup_steps)
                    )
                    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                    lr = min_lr + (base_lr - min_lr) * cosine_factor
                elif self.decay_strategy == "step":
                    # Step decay after warmup
                    decay_steps = current_step - self.warmup_steps
                    num_decays = decay_steps // self.step_size
                    lr = base_lr * (self.step_gamma**num_decays)
                    lr = max(lr, min_lr)  # Don't go below min_lr

            lrs.append(lr)

        return lrs

    def step(self, epoch=None):
        """Override step to ensure proper step counting."""
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the scheduler."""
        return {
            "base_lrs_dict": self.base_lrs_dict,
            "group_names": self.group_names,
            "last_step": self.last_epoch,  # Store as last_step for clarity
            "min_lr_ratios": self.min_lr_ratios,
            "warmup_start_lr_ratios": self.warmup_start_lr_ratios,
            "_step_count": self._step_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the schedulers state."""
        self.base_lrs_dict = state_dict["base_lrs_dict"]
        self.group_names = state_dict["group_names"]
        self.last_epoch = state_dict["last_step"]  # Load as last_epoch for parent class
        self.min_lr_ratios = state_dict["min_lr_ratios"]
        self.warmup_start_lr_ratios = state_dict["warmup_start_lr_ratios"]
        self._step_count = state_dict["_step_count"]


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, train_cfg: Dict[str, Any], total_steps: int
) -> Optional[_LRScheduler]:
    """Build scheduler with configuration for backbone and transformer."""
    scheduler_cfg = train_cfg.get("scheduler", {})
    if not scheduler_cfg or not scheduler_cfg.get("type"):
        return None

    scheduler_type = scheduler_cfg["type"].lower()

    if scheduler_type == "warmup":

        min_lr_ratios = {}
        warmup_start_ratios = {}

        if scheduler_cfg.get("backbone_scheduler_enabled", False):
            min_lr_ratios["backbone"] = scheduler_cfg.get("backbone_min_lr_ratio", 0.1)
            warmup_start_ratios["backbone"] = scheduler_cfg.get(
                "backbone_warmup_start_ratio", 0.01
            )
        else:
            min_lr_ratios["backbone"] = 1.0
            warmup_start_ratios["backbone"] = 1.0

        if scheduler_cfg.get("transformer_scheduler_enabled", True):
            min_lr_ratios["transformer"] = scheduler_cfg.get(
                "transformer_min_lr_ratio", 0.1
            )
            warmup_start_ratios["transformer"] = scheduler_cfg.get(
                "transformer_warmup_start_ratio", 0.01
            )
        else:
            min_lr_ratios["transformer"] = 1.0
            warmup_start_ratios["transformer"] = 1.0

        return MultiGroupWarmupScheduler(
            optimizer,
            warmup_steps=scheduler_cfg.get("warmup_steps", 500),
            total_steps=total_steps,
            decay_strategy=scheduler_cfg.get("decay_strategy", "cosine"),
            min_lr_ratios=min_lr_ratios,
            warmup_start_lr_ratios=warmup_start_ratios,
            step_size=scheduler_cfg.get("step_size"),
            step_gamma=scheduler_cfg.get("step_gamma"),
        )

    raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class KLLossWeightScheduler:
    def __init__(
        self,
        schedule_type="linear",
        min_weight=0.0,
        max_weight=1.0,
        total_annealing_steps=None,
        M=None,
        R=None,
    ):
        self.schedule_type = schedule_type
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.current_step = 0
        if self.schedule_type == "cyclic":
            if M is None or R is None:
                raise ValueError("M and R must be specified for cyclic kl_scheduler")
            self.M = M
            self.R = R
            self.cycle_steps = M
            self.ramp_up_steps = int(R * M)
        elif self.schedule_type == "linear":
            if total_annealing_steps is None:
                raise ValueError(
                    "total_annealing_steps must be specified for linear kl_scheduler"
                )
            self.total_annealing_steps = total_annealing_steps
        else:
            raise ValueError(f"Unknown kl_scheduler type: {self.schedule_type}")

    def get_kl_loss_weight(self):
        if self.schedule_type == "linear":
            if self.current_step >= self.total_annealing_steps:
                return self.max_weight
            else:
                progress = self.current_step / self.total_annealing_steps
                kl_weight = (
                    self.min_weight + (self.max_weight - self.min_weight) * progress
                )
                return kl_weight
        elif self.schedule_type == "cyclic":
            step_in_cycle = self.current_step % self.M
            if step_in_cycle < self.ramp_up_steps:
                # Linear increase
                progress = step_in_cycle / self.ramp_up_steps
                kl_weight = (
                    self.min_weight + (self.max_weight - self.min_weight) * progress
                )
            else:
                kl_weight = self.max_weight
            return kl_weight
        else:
            return self.max_weight  # default

    def step(self):
        self.current_step += 1


def build_kl_scheduler(train_cfg, total_steps, steps_per_epoch):

    kl_scheduler_cfg = train_cfg.get("kl_scheduler", {})

    # if kl_scheduler is not enable return if None
    if not kl_scheduler_cfg.get("kl_scheduler_enabled", False):
        return None

    kl_scheduler_type = kl_scheduler_cfg.get("type", "linear")
    min_kl_weight = kl_scheduler_cfg.get("min_kl_loss_weight", 0.0)
    max_kl_weight = kl_scheduler_cfg.get("max_kl_loss_weight", 1.0)
    if kl_scheduler_type == "linear":
        annealing_epochs = kl_scheduler_cfg.get(
            "annealing_epoch", total_steps // steps_per_epoch
        )
        return KLLossWeightScheduler(
            schedule_type="linear",
            min_weight=min_kl_weight,
            max_weight=max_kl_weight,
            total_annealing_steps=annealing_epochs * steps_per_epoch,
        )
    elif kl_scheduler_type == "cyclic":
        num_epochs_per_cycle = kl_scheduler_cfg.get("num_epochs_per_cycle")
        M = num_epochs_per_cycle * steps_per_epoch
        R = kl_scheduler_cfg.get("R")
        if M is None or R is None:
            raise ValueError("For cyclic kl_scheduler, 'M' and 'R' must be specified.")
        return KLLossWeightScheduler(
            schedule_type="cyclic",
            min_weight=min_kl_weight,
            max_weight=max_kl_weight,
            M=M,
            R=R,
        )
    else:
        raise ValueError(f"Unknown kl_scheduler type: {kl_scheduler_type}")
