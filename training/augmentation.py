import random
import torch
import logging
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Optional, Any, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for image augmentations."""

    enabled: bool = True
    size: Tuple[int, int] = (224, 224)
    normalize: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
    )
    cameras: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    interpolation: str = "bilinear"

    def __post_init__(self):
        if isinstance(self.size, list):
            self.size = tuple(self.size)
        if self.cameras is None:
            self.cameras = {}


class BaseTransform:
    """Default transform applied when no augmentations are used."""

    def __init__(self, config: Union[AugmentationConfig, Dict[str, Any], None] = None):
        if config is None:
            config = AugmentationConfig()
        elif isinstance(config, dict):
            config = AugmentationConfig(**config)

        interpolation_method = (
            F.InterpolationMode.BILINEAR
            if config.interpolation == "bilinear"
            else F.InterpolationMode.NEAREST
        )

        self.transform = T.Compose(
            [
                T.Resize(size=config.size, interpolation=interpolation_method),
                T.Normalize(mean=config.normalize["mean"], std=config.normalize["std"]),
            ]
        )

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.transform(image)


class SequenceConsistentAugmentation:
    """Applies consistent augmentations across a sequence of images."""

    def __init__(
        self, camera_name: str, aug_config: Dict[str, Any], seed: Optional[int] = None
    ):
        self.camera_name = camera_name
        self.aug_config = aug_config
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.transform = self._build_transform()

    def _build_transform(self) -> T.Compose:
        """Build the torchvision transform pipeline."""
        transforms_list = []

        # Always add resize first
        interpolation_method = (
            F.InterpolationMode.BILINEAR
            if self.aug_config.get("interpolation", "bilinear") == "bilinear"
            else F.InterpolationMode.NEAREST
        )
        transforms_list.append(
            T.Resize(size=self.aug_config["size"], interpolation=interpolation_method)
        )

        # Add augmentations if enabled
        if self.aug_config["enabled"]:
            for aug_name, params in self.aug_config.get("augmentations", {}).items():
                transform = self._get_transform(aug_name, params)
                if transform:
                    transforms_list.append(transform)

        # Always add normalization
        transforms_list.append(
            T.Normalize(
                mean=self.aug_config["normalize"]["mean"],
                std=self.aug_config["normalize"]["std"],
            )
        )

        return T.Compose(transforms_list)

    def _get_transform(self, aug_name: str, params: Dict[str, Any]):
        """Map augmentation name to torchvision transform."""
        if aug_name == "RandomBrightnessContrast":
            brightness = params.get("brightness_limit", [0.0, 0.0])
            contrast = params.get("contrast_limit", [0.0, 0.0])
            brightness_factor = brightness[1] - brightness[0]
            contrast_factor = contrast[1] - contrast[0]
            p = params.get("p", 0.5)
            return T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=brightness_factor,
                        contrast=contrast_factor,
                        saturation=0,  # Default to 0 since not specified in the config
                        hue=0,  # Default to 0 since not specified in the config
                    )
                ],
                p=p,
            )
        elif aug_name == "GaussianBlur":
            kernel_size = params.get("blur_limit", [3, 3])
            sigma = params.get("sigma_limit", [0.1, 2.0])
            p = params.get("p", 0.5)
            return T.RandomApply(
                [T.GaussianBlur(kernel_size=max(kernel_size), sigma=sigma)], p=p
            )
        elif aug_name == "Affine":
            scale = params.get("scale", [1.0, 1.0])
            translate_percent = params.get("translate_percent", [0.0, 0.0])
            degrees = params.get("rotate", [0, 0])
            p = params.get("p", 0.5)
            return T.RandomApply(
                [
                    T.RandomAffine(
                        degrees=degrees,
                        translate=tuple(translate_percent),
                        scale=tuple(scale),
                    )
                ],
                p=p,
            )
        elif aug_name == "HueSaturationValue":
            hue = params.get("hue_shift_limit", 0)
            saturation = params.get("sat_shift_limit", 0)
            p = params.get("p", 0.5)
            return T.RandomApply(
                [
                    T.ColorJitter(
                        hue=hue,
                        saturation=saturation,
                        brightness=0,  # Default to 0 since not specified in the config
                        contrast=0,  # Default to 0 since not specified in the config
                    )
                ],
                p=p,
            )
        elif aug_name == "Rotate":
            degrees = params.get("limit", [0, 0])
            p = params.get("p", 0.5)
            return T.RandomApply([T.RandomRotation(degrees=degrees)], p=p)
        else:
            logger.warning(f"Augmentation {aug_name} not supported")
            return None

    def __call__(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply consistent augmentation to a sequence of images."""
        transformed_images = []

        # Save the current random state
        rng_state = random.getstate()
        torch_rng_state = torch.get_rng_state()
        np_rng_state = np.random.get_state()

        for image in images:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

            img = self.transform(image)
            transformed_images.append(img)

        # Restore the random state
        random.setstate(rng_state)
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(np_rng_state)
        return transformed_images


class ImageAugmentationPipeline:
    """Manages augmentations for all cameras and sequences."""

    def __init__(self, cfg: Union[DictConfig, Dict[str, Any]]):
        try:
            # Convert to dict for consistent handling
            if isinstance(cfg, DictConfig):
                cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            else:
                cfg_dict = cfg

            # Get image_augmentation config with defaults
            aug_config = cfg_dict.get("image_augmentation", {})

            # Set default values
            aug_config.setdefault("enabled", True)
            aug_config.setdefault("size", (224, 224))
            aug_config.setdefault("cameras", {})
            aug_config.setdefault(
                "normalize",
                {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            )
            aug_config.setdefault("interpolation", "bilinear")

            self.cfg = AugmentationConfig(**aug_config)
            self.base_transform = BaseTransform(self.cfg)

        except Exception as e:
            logger.error(f"Error initializing augmentation config: {str(e)}")
            raise

    def get_sequence_transform(
        self, camera_name: str, seed: Optional[int] = None
    ) -> Optional[SequenceConsistentAugmentation]:
        """Get a sequence-consistent transform for a camera."""
        if not self.cfg.enabled or camera_name not in self.cfg.cameras:
            return None

        camera_config = self.cfg.cameras[camera_name]
        if not camera_config.get("enabled", True):
            return None

        try:
            # Merge global config into camera config
            merged_config = {**camera_config}
            merged_config.setdefault("normalize", self.cfg.normalize)
            merged_config.setdefault("size", self.cfg.size)
            merged_config.setdefault("interpolation", self.cfg.interpolation)
            merged_config.setdefault("enabled", self.cfg.enabled)

            return SequenceConsistentAugmentation(
                camera_name=camera_name, aug_config=merged_config, seed=seed
            )
        except Exception as e:
            logger.error(f"Error creating transform for camera {camera_name}: {str(e)}")
            return None

    def get_base_transform(self) -> BaseTransform:
        """Get the base transform for images without augmentation."""
        return self.base_transform


def setup_augmentation_pipeline(
    cfg: Optional[DictConfig] = None,
) -> Optional[ImageAugmentationPipeline]:
    """Factory function to create augmentation pipeline."""
    if cfg is None:
        # Create default config
        cfg = {
            "image_augmentation": {
                "enabled": True,
                "size": (224, 224),
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                "cameras": {},  # Empty dict for no camera-specific augmentations
                "interpolation": "bilinear",
            }
        }

    try:
        return ImageAugmentationPipeline(cfg)
    except Exception as e:
        logger.error(f"Error setting up augmentation pipeline: {str(e)}")
        return None
