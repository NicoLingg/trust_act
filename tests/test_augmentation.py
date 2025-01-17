# tests/test_augmentation.py
import sys
import os
import torch
import pytest
import shutil
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from training.dataloader import load_data
from training.types import IMAGE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAugmentation:
    def __init__(self, dataset_path: str = "data/stacking_v3_processed_224px_s1"):
        """Initialize test class with dataset path.

        Args:
            dataset_path: Path to the dataset directory
        """
        self.dataset_path = dataset_path
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")

        self.output_dir = Path("test_output")
        self.output_dir.mkdir(exist_ok=True)
        self.aug_config = OmegaConf.load("config/augmentation_config.yaml")

        logger.info(f"Using dataset path: {self.dataset_path}")

    @classmethod
    def teardown_class(cls):
        """Cleanup after all tests."""
        if cls.output_dir.exists():
            shutil.rmtree(cls.output_dir)

    def visualize_sequence(self, images, save_path, title, camera_name):
        """Visualize a sequence of images.

        Args:
            images: Tensor of shape [sequence_length, C, H, W]
        """
        if len(images.shape) != 4:
            raise ValueError(
                f"Expected images tensor of shape [seq_len, C, H, W], got {images.shape}"
            )

        num_images = images.shape[0]  # sequence length
        fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 4))

        if num_images == 1:
            axes = [axes]

        # Create a suptitle with camera name and sequence length
        plt.suptitle(f"{title}\nCamera: {camera_name}, Sequence Length: {num_images}")

        for i, ax in enumerate(axes):
            # Convert tensor to numpy and denormalize
            img = images[i].cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)

            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"Frame {i}")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def get_dataloader(self, dataset_path: str, hist_steps: int, hist_stride: int = 1):
        """Helper to create a dataloader with specific parameters."""
        self.dataset_path = dataset_path
        return load_data(
            dataset_dir=dataset_path,
            camera_names=["wrist_rgb", "base_rgb"],
            batch_size_train=1,
            batch_size_val=1,
            num_action_steps=1,
            num_hist_steps=hist_steps,
            hist_stride=hist_stride,
            sampling_stride=1,
            num_workers=1,
            augmentation_cfg=self.aug_config,
        )[
            0
        ]  # Return only train loader

    def test_different_sequence_lengths(self):
        """Test different sequence lengths and strides."""
        test_params = [
            {"num_hist_steps": 1, "hist_stride": 1, "name": "single_frame"},
            {"num_hist_steps": 5, "hist_stride": 1, "name": "sequence_5"},
            {"num_hist_steps": 3, "hist_stride": 2, "name": "sequence_3_stride_2"},
        ]

        for params in test_params:
            logger.info(f"Testing configuration: {params['name']}")

            try:
                train_loader = self.get_dataloader(
                    self.dataset_path, params["num_hist_steps"], params["hist_stride"]
                )

                # Get multiple batches to verify consistency
                for batch_idx, batch in enumerate(train_loader):
                    # batch[IMAGE] shape: [B, seq_len, num_cameras, C, H, W]
                    images = batch[IMAGE][
                        0
                    ]  # Remove batch dimension -> [seq_len, num_cameras, C, H, W]

                    # For each camera
                    for cam_idx, camera in enumerate(["wrist_rgb", "base_rgb"]):
                        cam_images = images[:, cam_idx]  # [seq_len, C, H, W]
                        save_path = (
                            self.output_dir
                            / f"sequence_{camera}_{params['name']}_batch{batch_idx}.png"
                        )

                        self.visualize_sequence(
                            cam_images, save_path, f"Test: {params['name']}", camera
                        )
                        logger.info(f"Saved visualization to {save_path}")

                    if batch_idx >= 2:  # Limit to 3 batches
                        break

            except Exception as e:
                logger.error(f"Error testing {params['name']}: {str(e)}")
                raise

    def test_augmentation_consistency(self):
        """Test that augmentations are consistent within sequences."""
        hist_steps = 1
        train_loader = self.get_dataloader(self.dataset_path, hist_steps)

        # Get a batch
        batch = next(iter(train_loader))
        images = batch[IMAGE][0]  # [seq_len, num_cameras, C, H, W]

        for cam_idx, camera in enumerate(["wrist_rgb", "base_rgb"]):
            cam_images = images[:, cam_idx]  # [seq_len, C, H, W]

            # Check that augmentation parameters are consistent across sequence
            diffs = []
            for i in range(1, cam_images.shape[0]):  # iterate over sequence length
                diff = torch.mean(torch.abs(cam_images[i] - cam_images[i - 1]))
                diffs.append(diff.item())

            # Verify differences are minimal, indicating consistency
            if len(diffs) > 0:  # only if sequence length > 1
                diffs = np.array(diffs)
                std_diff = np.std(diffs)
                logger.info(
                    f"Standard deviation of frame differences for {camera}: {std_diff}"
                )
                assert (
                    std_diff < 0.1
                ), f"Large variation in frame differences for {camera}: {std_diff}"


if __name__ == "__main__":
    dataset_path = "data/stacking_v3_processed_224px_s1"
    # Run tests
    test = TestAugmentation(dataset_path)
    try:
        test.test_different_sequence_lengths()
        test.test_augmentation_consistency()
        logger.info("All tests passed successfully!")
    except Exception as e:
        logger.error(f"Tests failed: {str(e)}")
        raise
    # finally:
    #     test.teardown_class() # Uncomment this line to cleanup test output directory
