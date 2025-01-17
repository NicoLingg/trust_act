import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hydra
from omegaconf import DictConfig
from training.utils import set_seed
from training.dataloader import load_data
from training.trainer import Trainer


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.train.seed)

    # If you have any auxiliary targets specified in the policy
    aux_targets = None
    if len(cfg.policy.action_aux_loss) > 0:
        aux_targets = list(cfg.policy.action_aux_loss.keys())

    train_dataloader, val_dataloader = load_data(
        dataset_dir=cfg.task.dataset_dir,
        camera_names=cfg.policy.camera_names,
        batch_size_train=cfg.data.batch_size_train,
        batch_size_val=cfg.data.batch_size_val,
        num_action_steps=cfg.policy.num_action_steps,
        num_hist_steps=cfg.policy.num_hist_steps,
        hist_stride=cfg.policy.hist_stride,
        sampling_stride=cfg.policy.sampling_stride,
        num_workers=cfg.data.num_workers_batch,
        prefetch_factor=cfg.data.prefetch_factor,
        train_ratio=cfg.data.train_split,
        aux_targets=aux_targets,
        augmentation_cfg=cfg.augmentation,
        balance_trust_levels=cfg.data.balance_trust_levels,
    )

    trainer = Trainer(cfg, train_dataloader, val_dataloader)
    trainer.train_bc()


if __name__ == "__main__":
    main()
