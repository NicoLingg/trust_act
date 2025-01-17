import os
import gc
import torch


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    val_loss: float = None,
    is_best: bool = False,
    is_last: bool = False,
    cfg=None,
):
    """Save checkpoint with scheduler state."""
    if is_best:
        ckpt_name = f"policy_best_{int(epoch)}.ckpt"
    elif is_last:
        ckpt_name = f"policy_last.ckpt"
    else:
        ckpt_name = f"policy_epoch_{int(epoch)}.ckpt"

    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": cfg,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, ckpt_path)
    print(f"Saved checkpoint at epoch {epoch}")


def clear_memory_cache():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
