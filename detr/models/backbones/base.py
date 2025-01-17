import torch
import torch.nn as nn
from typing import Tuple, Optional
from ..position_encoding import build_position_encoding


class BackboneBase(nn.Module):
    """Base backbone with position embeddings"""

    def __init__(self, args):
        super().__init__()
        self._model = None
        self._num_channels = None
        self._pos_embed = build_position_encoding(args)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning features and position embeddings"""
        features = self.get_features(x)
        pos = self._pos_embed(features)
        return features, pos

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features"""
        raise NotImplementedError

    def freeze(self):
        """Freeze backbone parameters"""
        for param in self._model.parameters():
            param.requires_grad_(False)

    @property
    def num_channels(self):
        return self._num_channels
