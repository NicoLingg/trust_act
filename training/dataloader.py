import os
import torch
import random
import numpy as np
import torchvision.io as io
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Any

from training.types import ACTIONS, IMAGE, ROBOT_STATE, TRUST
from training.augmentation import setup_augmentation_pipeline


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class RoboticDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        cameras: List[str],
        num_action_steps: int,
        num_hist_steps: int = 1,
        split: str = "train",
        train_ratio: float = 0.8,
        aux_targets: List[str] = None,
        sampling_stride: int = 1,
        hist_stride: int = 1,
        augmentation_cfg: Optional[DictConfig] = None,
        balance_trust_levels: bool = False,
    ):
        self.dataset_dir = dataset_dir
        self.num_action_steps = num_action_steps
        self.num_hist_steps = num_hist_steps
        self.cameras = cameras
        self.aux_targets = aux_targets
        self.split = split
        self.train_ratio = train_ratio
        self.balance_trust_levels = balance_trust_levels

        # Setup augmentation pipeline
        self.augmentation_pipeline = setup_augmentation_pipeline(
            augmentation_cfg if augmentation_cfg else {"image_augmentation": {}}
        )

        # Validate strides
        assert (
            hist_stride % sampling_stride == 0
        ), "Hist stride must be multiple of sampling stride"
        self.hist_stride = hist_stride
        self.stride = sampling_stride
        self.episode_dirs = self._setup_episode_dirs(split, train_ratio)
        self.samples, self.sequence_lengths = self._build_dataset_index()

        self.sequence_cache: Dict[int, Any] = {}

    def _setup_episode_dirs(self, split: str, train_ratio: float) -> List[str]:
        """Setup and split episode directories."""
        episode_dirs = [
            d
            for d in os.listdir(self.dataset_dir)
            if os.path.isdir(os.path.join(self.dataset_dir, d))
            and d.startswith("episode_")
        ]
        # Sort episodes to ensure deterministic ordering before shuffling
        episode_dirs.sort()
        rng = np.random.RandomState(42)
        rng.shuffle(episode_dirs)
        # episode_dirs = episode_dirs[:5]
        num_episodes = len(episode_dirs)
        print(f"Found {num_episodes} episodes in {self.dataset_dir}")

        # Split into train and validation
        num_train = int(train_ratio * num_episodes)
        if split == "train":
            return episode_dirs[:num_train]
        elif split == "val":
            return episode_dirs[num_train:]
        raise ValueError(f"Invalid split: {split}")

    def _build_dataset_index(self) -> Tuple[List[Tuple[int, int]], Dict[int, int]]:
        """Build the dataset index with samples and sequence lengths."""
        samples = []
        sequence_lengths = {}
        trust_categories = {"low": [], "medium": [], "high": []}

        print("Building dataset index...")
        for episode_dir_name in tqdm(self.episode_dirs, desc=f"Indexing episodes"):
            episode_id = int(episode_dir_name.replace("episode_", ""))
            sequence_path = os.path.join(
                self.dataset_dir, episode_dir_name, "sequence.npz"
            )

            # Load sequence data
            with np.load(sequence_path) as seq_data:
                seq_length = len(seq_data["actions"])
                sequence_lengths[episode_id] = seq_length
                trust_levels = seq_data["trust_levels"]
                trust_levels = np.clip(trust_levels, 0.0, 1.0)

                # Generate valid start indices
                num_segments = (seq_length - self.num_action_steps) // self.stride + 1
                for i in range(num_segments):
                    start_index = i * self.stride
                    if start_index + self.num_action_steps <= seq_length:
                        # Get mean trust for this segment
                        segment_trust = trust_levels[
                            start_index : start_index + self.num_action_steps
                        ].mean()

                        # Categorize the segment
                        if segment_trust < 0.33:
                            trust_categories["low"].append((episode_id, start_index))
                        elif segment_trust < 0.66:
                            trust_categories["medium"].append((episode_id, start_index))
                        else:
                            trust_categories["high"].append((episode_id, start_index))

        # Print original distribution
        print("\nOriginal distribution:")
        total = sum(len(samples) for samples in trust_categories.values())
        for category, category_samples in trust_categories.items():
            print(
                f"{category}: {len(category_samples)} ({len(category_samples)/total*100:.1f}%)"
            )

        if self.balance_trust_levels:
            # Find minimum category size
            min_size = min(len(samples) for samples in trust_categories.values())

            # Balance categories
            balanced_samples = []
            for category_samples in trust_categories.values():
                selected = random.sample(category_samples, min_size)
                balanced_samples.extend(selected)
            samples = balanced_samples

            # Print balanced distribution
            print("\nBalanced distribution:")
            total = len(samples)
            balanced_cats = {"low": 0, "medium": 0, "high": 0}
            for episode_id, start_index in samples:
                with np.load(
                    os.path.join(
                        self.dataset_dir, f"episode_{episode_id}", "sequence.npz"
                    )
                ) as data:
                    trust = data["trust_levels"][
                        start_index : start_index + self.num_action_steps
                    ].mean()
                    if trust < 0.33:
                        balanced_cats["low"] += 1
                    elif trust < 0.66:
                        balanced_cats["medium"] += 1
                    else:
                        balanced_cats["high"] += 1

            for category, count in balanced_cats.items():
                print(f"{category}: {count} ({count/total*100:.1f}%)")
        else:
            # Use all samples without balancing
            for category_samples in trust_categories.values():
                samples.extend(category_samples)

        # Shuffle samples
        random.shuffle(samples)
        return samples, sequence_lengths

    @staticmethod
    def _get_sequence_length(sequence_path: str) -> int:
        """Get sequence length from file."""
        with np.load(sequence_path, mmap_mode="r") as sequence_data:
            return sequence_data["actions"].shape[0]

    def _get_sequence_data(self, episode_id: int):
        """Cache sequence data for faster access."""
        if episode_id not in self.sequence_cache:
            sequence_path = os.path.join(
                self.dataset_dir, f"episode_{episode_id}", "sequence.npz"
            )
            self.sequence_cache[episode_id] = np.load(sequence_path, allow_pickle=True)
        return self.sequence_cache[episode_id]

    def __len__(self) -> int:
        return len(self.samples)

    def load_single_image(self, image_path: str) -> torch.Tensor:
        """Load a single image as a tensor."""
        image = io.read_image(image_path)  # Shape: [C, H, W]
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = image.float() / 255.0  # Normalize to [0, 1]
        return image

    def process_image_sequence(
        self, images: List[torch.Tensor], sequence_transform: Optional[Any]
    ) -> List[torch.Tensor]:
        """Process a sequence of images with consistent transforms."""
        if sequence_transform is not None:
            # Apply sequence-consistent augmentation
            return sequence_transform(images)
        else:
            # Apply base transform to each image independently
            base_transform = self.augmentation_pipeline.get_base_transform()
            processed_images = []
            for image in images:
                processed = base_transform(image)
                processed_images.append(processed)
            return processed_images

    def get_history_indices(self, start: int) -> List[int]:
        """Calculate history indices with proper handling of edge cases."""
        hist_stride = self.hist_stride
        hist_start = max(0, start - (self.num_hist_steps - 1) * hist_stride)

        # Generate indices from newest to oldest
        indices = list(range(start, hist_start - 1, -hist_stride))
        indices = indices[::-1]  # Reverse to get chronological order

        # Ensure we have exactly num_hist_steps indices
        if len(indices) < self.num_hist_steps:
            # Pad with duplicates of the earliest frame if needed
            padding = self.num_hist_steps - len(indices)
            indices = [indices[0]] * padding + indices

        return indices[: self.num_hist_steps]  # Ensure we don't exceed num_hist_steps

    def __getitem__(self, index: int) -> Dict[str, Any]:
        episode_id, start = self.samples[index]
        sequence_data = self._get_sequence_data(episode_id)

        # Get actions and trust values
        actions = sequence_data["actions"][
            start : start + self.num_action_steps
        ].astype(np.float32)
        trust = sequence_data["trust_levels"][
            start : start + self.num_action_steps
        ].astype(np.float32)
        trust = np.clip(trust, 0.0, 1.0)

        # Get history indicess
        hist_indices = self.get_history_indices(start)

        # Get robot state
        qpos = sequence_data["joint_positions"][hist_indices].astype(np.float32)

        # Initialize sequence-consistent augmentations for each camera
        sequence_transforms = {}
        if self.augmentation_pipeline is not None:
            # Use a random seed instead of one based on episode_id and start
            seed = random.randint(0, 2**32 - 1)
            for camera in self.cameras:
                transform = self.augmentation_pipeline.get_sequence_transform(
                    camera, seed=seed
                )
                if transform is not None:
                    sequence_transforms[camera] = transform

        # Process images for each camera
        camera_images = {}
        for camera in self.cameras:
            # Load all images for this camera's sequence
            raw_images = []
            for i in hist_indices:
                image_path = os.path.join(
                    self.dataset_dir,
                    f"episode_{episode_id}",
                    "frames",
                    f"{i:06d}",
                    f"{camera}.jpg",
                )
                raw_images.append(self.load_single_image(image_path))

            # Process entire sequence with consistent transforms
            sequence_transform = sequence_transforms.get(camera)
            processed_images = self.process_image_sequence(
                raw_images, sequence_transform
            )
            camera_images[camera] = torch.stack(
                processed_images
            )  # Shape: [seq_len, C, H, W]

        # Stack all camera sequences
        images = torch.stack(
            [camera_images[cam] for cam in self.cameras]
        )  # Shape: [num_cameras, seq_len, C, H, W]
        images = images.permute(1, 0, 2, 3, 4)  # [seq_len, num_cameras, C, H, W]

        return_dict = {
            IMAGE: images,
            ROBOT_STATE: torch.from_numpy(qpos),
            ACTIONS: torch.from_numpy(actions),
            TRUST: torch.from_numpy(trust),
        }

        if self.aux_targets is not None:
            for name in self.aux_targets:
                target = sequence_data[name][
                    start : start + self.num_action_steps
                ].astype(np.float32)
                return_dict[name] = torch.from_numpy(target)

        return return_dict


def load_data(
    dataset_dir: str,
    camera_names: List[str],
    batch_size_train: int,
    batch_size_val: int,
    num_action_steps: int,
    num_hist_steps: int,
    hist_stride: int,
    sampling_stride: int,
    num_workers: Optional[int] = None,
    prefetch_factor: int = 2,
    train_ratio: float = 0.8,
    aux_targets: List[str] = None,
    augmentation_cfg: Optional[DictConfig] = None,
    balance_trust_levels: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load training and validation data loaders.

    Args:
        dataset_dir: Path to dataset directory
        camera_names: List of camera names to use
        batch_size_train: Batch size for training
        batch_size_val: Batch size for validation
        num_action_steps: Number of action steps to predict
        num_hist_steps: Maximum number of history steps to use
        hist_stride: Stride for historical data
        sampling_stride: Sampling stride for data
        num_workers: Number of workers for data loading
        prefetch_factor: Number of batches to prefetch
        train_ratio: Ratio of data to use for training
        aux_targets: List of auxiliary targets
        augmentation_cfg: Augmentation configuration

    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    train_dataset = RoboticDataset(
        dataset_dir=dataset_dir,
        num_action_steps=num_action_steps,
        num_hist_steps=num_hist_steps,
        hist_stride=hist_stride,
        sampling_stride=sampling_stride,
        cameras=camera_names,
        split="train",
        train_ratio=train_ratio,
        aux_targets=aux_targets,
        augmentation_cfg=augmentation_cfg,
        balance_trust_levels=balance_trust_levels,
    )

    val_dataset = RoboticDataset(
        dataset_dir=dataset_dir,
        num_action_steps=num_action_steps,
        num_hist_steps=num_hist_steps,
        hist_stride=hist_stride,
        sampling_stride=sampling_stride,
        cameras=camera_names,
        split="val",
        train_ratio=train_ratio,
        aux_targets=aux_targets,
        augmentation_cfg=augmentation_cfg,
        balance_trust_levels=False,
    )

    num_workers = os.cpu_count() if num_workers is None else int(num_workers)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,  # Ensure different seeds in each worker
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader
