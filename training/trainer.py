import gc
import os
import yaml
import time
import wandb
import torch
import shutil

from typing import Dict, List, Optional, Union
from training.optimizers import build_optimizer_and_schedulers
from training.utils import format_time, compute_dict_mean, format_loss_string
from training.callbacks import save_checkpoint, clear_memory_cache
from training.policy import ACTPolicy
from training.types import (
    STANDARD_INPUTS,
    TrustModelling,
    TrustRewardSetting,
    TrustRewardReduction,
)


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return None


class Trainer:
    def __init__(self, cfg, train_dataloader, val_dataloader):
        self.cfg = cfg
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = (
            self.cfg.train.total_steps
            if self.cfg.train.total_steps is not None
            else self.steps_per_epoch * self.cfg.train.num_epochs
        )

        run = wandb.init(
            mode="online",
            project=cfg.task.name,
            config={
                "task": cfg.task,
                "train": cfg.train,
                "policy": cfg.policy,
            },
        )

        self._setup_environment()
        self.policy = self._setup_policy()

        self.optimizer, self.scheduler, self.kl_scheduler = (
            build_optimizer_and_schedulers(
                self.policy, cfg.train, self.total_steps, self.steps_per_epoch
            )
        )

        # Set the scheduler in the policy
        if self.kl_scheduler is not None:
            self.policy.set_kl_scheduler(self.kl_scheduler)

        self.checkpoint_dir = os.path.join(
            cfg.train.checkpoint_dir,
            f"{cfg.task.name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}_{run.name}",
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Copy configs
        self._copy_config_files()

        # Save dataset splits
        self._save_dataset_splits(train_dataloader.dataset, val_dataloader.dataset)

        # Initialize training state
        self.current_step = 0
        self.steps_per_epoch = None
        self.best_val_loss = float("inf")
        self.train_history = []
        self.validation_history = []

        # Step-based frequencies (will be set in train_bc)
        self.log_freq_steps = None
        self.val_freq_steps = None
        self.checkpoint_steps = None

        # For timing and progress tracking
        self.step_times = []
        self.step_time_window = 100
        self.step_start_time = None

        # Setup AMP scaler
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

    def _save_dataset_splits(self, train_dataset, val_dataset):
        """Save dataset splits to checkpoint directory as YAML."""
        splits_file = os.path.join(self.checkpoint_dir, "dataset_splits.yaml")
        splits_data = {
            "train_episodes": train_dataset.episode_dirs,
            "val_episodes": val_dataset.episode_dirs,
            "train_ratio": train_dataset.train_ratio,
            "total_episodes": len(train_dataset.episode_dirs)
            + len(val_dataset.episode_dirs),
            "dataset_dir": train_dataset.dataset_dir,
        }

        with open(splits_file, "w") as f:
            yaml.safe_dump(splits_data, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Saved dataset splits to {splits_file}")

    def _setup_environment(self):
        """Configure PyTorch training environment and optimizations."""
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()

            # Log GPU info
            gpu_name = torch.cuda.get_device_name()
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ Using {gpu_name} ({memory_gb:.1f}GB)")
            print("✓ Enabled automatic mixed precision")
        else:
            print("⚠ Running on CPU - disabled GPU optimizations")

    def _setup_policy(self):
        """Initialize and configure the policy network."""
        self.cfg.policy.trust_modelling = TrustModelling[
            self.cfg.policy.trust_modelling
        ]
        self.cfg.inference.trust_reward_setting = TrustRewardSetting[
            self.cfg.inference.trust_reward_setting
        ]
        self.cfg.inference.trust_reward_reduction = TrustRewardReduction[
            self.cfg.inference.trust_reward_reduction
        ]

        policy = ACTPolicy(self.cfg).to(self.device)

        if torch.cuda.is_available() and hasattr(torch, "compile"):
            try:
                print("⚡ Compiling model for acceleration...")
                # policy = torch.compile(policy, fullgraph=True)
                print("✓ Model compilation successful")
            except Exception as e:
                print(f"⚠ Model compilation failed: {str(e)}")
                print("  → Continuing with uncompiled model")
        return policy

    def _copy_config_files(self):
        """Copy all configuration files from the config directory to checkpoint/config directory."""
        # Create config subdirectory in checkpoint directory
        checkpoint_config_dir = os.path.join(self.checkpoint_dir, "config")
        os.makedirs(checkpoint_config_dir, exist_ok=True)

        # Copy stats.yaml
        stats_src = os.path.join(self.cfg.task.dataset_dir, "stats.yaml")
        stats_dst = os.path.join(
            self.checkpoint_dir, "stats.yaml"
        )  # Keep stats.yaml in main checkpoint dir
        if os.path.exists(stats_src):
            shutil.copyfile(stats_src, stats_dst)
            print(f"✓ Copied stats.yaml to {self.checkpoint_dir}")
        else:
            print(f"⚠ stats.yaml not found in {stats_src}")

        # Copy all yaml files from config directory to checkpoint/config
        config_dir = "config"
        if os.path.exists(config_dir):
            for config_file in os.listdir(config_dir):
                if config_file.endswith(".yaml"):
                    src = os.path.join(config_dir, config_file)
                    dst = os.path.join(checkpoint_config_dir, config_file)
                    shutil.copyfile(src, dst)
                    print(f"✓ Copied {config_file} to {checkpoint_config_dir}")
        else:
            print(f"⚠ Config directory not found at {config_dir}")

    def _prepare_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> List[Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        data = []
        for k in STANDARD_INPUTS:
            data.append(batch[k] if k in batch else None)

        # data = [batch[k] for k in STANDARD_INPUTS if k in batch else None]
        aux_targets = {k: batch[k] for k in batch if k not in STANDARD_INPUTS}
        data.append(aux_targets)
        return data

    def _process_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process a single batch with automatic mixed precision."""
        self.optimizer.zero_grad(set_to_none=True)

        # Move data to device efficiently
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        batch = self._prepare_batch(batch)

        with torch.amp.autocast("cuda") if self.use_amp else nullcontext():
            loss_dict, latent_dict = self.policy(*batch)
            loss = loss_dict["loss"]

        # Backward pass with gradient scaling
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        if self.kl_scheduler is not None:
            self.kl_scheduler.step()

        return {k: v.detach() for k, v in loss_dict.items()}, {
            k: v.detach() for k, v in latent_dict.items()
        }

    def validate(self) -> Dict[str, torch.Tensor]:
        """Run validation pass."""
        self.policy.eval()
        val_losses = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = {
                    k: v.to(self.device, non_blocking=True) for k, v in batch.items()
                }
                batch = self._prepare_batch(batch)
                loss_dict, latent_dict = self.policy(*batch)
                val_losses.append({k: v.detach() for k, v in loss_dict.items()})

        self.policy.train()
        return compute_dict_mean(val_losses)

    def _configure_step_frequencies(self):
        """Configure all step-based frequencies based on dataset size."""

        # Get frequencies from config or use defaults
        log_freq = getattr(self.cfg.train, "log_step_frequency", self.steps_per_epoch)
        val_freq = getattr(
            self.cfg.train, "validation_step_frequency", self.steps_per_epoch * 5
        )
        checkpoint_freq = getattr(
            self.cfg.train, "checkpoint_step_frequency", self.steps_per_epoch * 10
        )

        # Set step frequencies
        self.log_freq_steps = log_freq
        self.val_freq_steps = val_freq
        self.checkpoint_steps = checkpoint_freq

        # Log configuration
        print(f"\nTraining Configuration:")
        print(f"Total steps: {self.total_steps:,}")
        print(f"Steps per epoch: {self.steps_per_epoch:,}")
        print(f"Log frequency: Every {self.log_freq_steps:,} steps")
        print(f"Validation frequency: Every {self.val_freq_steps:,} steps")
        print(f"Checkpoint frequency: Every {self.checkpoint_steps:,} steps")

    def _log_step_progress(
        self,
        train_summary: Dict[str, torch.Tensor],
        val_summary: Optional[Dict[str, torch.Tensor]],
        step_time: float,
        elapsed_time: float,
        eta: float,
        laten_dict,
    ):
        """Log training progress with step-based metrics."""
        current_epoch = self.current_step / self.steps_per_epoch

        metrics = {
            **{f"train/{k}": v.item() for k, v in train_summary.items()},
            "step": self.current_step,
            "epoch": current_epoch,
            "step_time": step_time,
            "total_time": elapsed_time,
        }

        # Log latent statistics (mu and logvar)
        # print(laten_dict)
        if laten_dict is not None:
            # Log mean statistics
            metrics["latent/mu_mean"] = laten_dict["mu"].mean().item()
            metrics["latent/mu_std"] = laten_dict["mu"].std().item()
            metrics["latent/mu_min"] = laten_dict["mu"].min().item()
            metrics["latent/mu_max"] = laten_dict["mu"].max().item()

            # Log variance statistics
            metrics["latent/logvar_mean"] = laten_dict["logvar"].mean().item()
            metrics["latent/logvar_std"] = laten_dict["logvar"].std().item()
            metrics["latent/logvar_min"] = laten_dict["logvar"].min().item()
            metrics["latent/logvar_max"] = laten_dict["logvar"].max().item()

            # Log per-dimension statistics
            for dim in range(laten_dict["mu"].size(1)):
                metrics[f"latent/mu_dim_{dim}"] = laten_dict["mu"][:, dim].mean().item()
                metrics[f"latent/logvar_dim_{dim}"] = (
                    laten_dict["logvar"][:, dim].mean().item()
                )

        # Add learning rates
        for group in self.optimizer.param_groups:
            metrics[f"lr/{group['name']}"] = group["lr"]

        # Add KL weight (will use default if scheduler is disabled)
        if self.cfg.policy.use_encoder:
            metrics["kl_weight"] = self.policy.get_kl_weight()

        if val_summary:
            metrics.update({f"val/{k}": v.item() for k, v in val_summary.items()})

        # Console logging
        print("-" * 50)
        print(
            f"Step {self.current_step:,}/{self.total_steps:,} (Epoch {current_epoch:.2f})"
        )
        print(format_loss_string(train_summary, "Train"))
        if val_summary:
            print(format_loss_string(val_summary, "Val  "))
        print(
            f"Learning rates: "
            + ", ".join(
                [
                    f"{group['name']}: {group['lr']:.2e}"
                    for group in self.optimizer.param_groups
                ]
            )
        )
        print(f"Total Time: {format_time(elapsed_time)} | ETA: {format_time(eta)}")

        wandb.log(metrics, step=self.current_step)

    def _memory_efficient_cycle(self):
        """Memory efficient version of itertools.cycle for DataLoader
        that doesn't retain references to previous iterations."""
        while True:
            iterator = iter(self.train_dataloader)
            while True:
                try:
                    yield next(iterator)
                except StopIteration:
                    del iterator
                    break
            gc.collect()

    def train_bc(self):
        """Continuous training loop with step-based logging."""
        self.steps_per_epoch = len(self.train_dataloader)
        self._configure_step_frequencies()

        total_start_time = time.time()
        self.step_start_time = time.time()
        train_losses = []

        train_iter = self._memory_efficient_cycle()

        try:
            while self.current_step < self.total_steps:
                batch = next(train_iter)
                loss_dict, laten_dict = self._process_batch(batch)
                train_losses.append(loss_dict)

                # Timing for this step
                step_time = time.time() - self.step_start_time
                self.step_times.append(step_time)
                if len(self.step_times) > self.step_time_window:
                    self.step_times.pop(0)
                self.step_start_time = time.time()

                # Progress tracking
                self.current_step += 1

                # Determine if we should log or validate
                should_log = (
                    self.current_step % self.log_freq_steps == 0
                    or self.current_step == self.total_steps
                )

                should_validate = (
                    self.current_step % self.val_freq_steps == 0
                    or self.current_step == self.total_steps
                )

                # First validation at step 10
                if self.current_step == 10:
                    should_log = should_validate = True

                if should_log or should_validate:
                    train_summary = compute_dict_mean(train_losses)

                    # Calculate timing
                    current_time = time.time()
                    elapsed_time = current_time - total_start_time
                    avg_step_time = sum(self.step_times) / len(self.step_times)
                    steps_remaining = self.total_steps - self.current_step
                    eta = avg_step_time * steps_remaining

                    # Run validation if needed
                    val_summary = self.validate() if should_validate else None

                    # Update best validation loss
                    is_best = False
                    if val_summary:
                        is_best = val_summary["loss"] < self.best_val_loss
                        if is_best:
                            self.best_val_loss = val_summary["loss"]

                    # Log progress
                    self._log_step_progress(
                        train_summary,
                        val_summary,
                        step_time,
                        elapsed_time,
                        eta,
                        laten_dict,
                    )

                # Handle checkpointing
                should_checkpoint = self.current_step % self.checkpoint_steps == 0
                if should_checkpoint or (should_validate and is_best):
                    current_epoch = self.current_step / self.steps_per_epoch
                    save_checkpoint(
                        self.checkpoint_dir,
                        current_epoch,
                        self.policy,
                        self.optimizer,
                        self.scheduler,
                        val_loss=val_summary["loss"] if val_summary else None,
                        is_best=should_validate and is_best,
                        cfg=self.cfg,
                    )
                    print(
                        f"Saved {'best ' if is_best else ''}checkpoint at step {self.current_step}"
                    )
                    clear_memory_cache()

                # Reset losses at epoch boundary
                train_losses = []

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")

        finally:
            # Save final model and cleanup
            final_val_summary = self.validate()
            final_epoch = self.current_step / self.steps_per_epoch
            save_checkpoint(
                self.checkpoint_dir,
                final_epoch,
                self.policy,
                self.optimizer,
                self.scheduler,
                val_loss=final_val_summary["loss"],
                is_last=True,
                cfg=self.cfg,
            )
            print(
                f"\nTraining completed in {format_time(time.time() - total_start_time)}"
            )
