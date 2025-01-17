import os
import torch
import torch.nn as nn
from torchvision import models
from torch.nn.modules.linear import Identity
from torchvision.models._utils import IntermediateLayerGetter


from detr.models.backbones.base import BackboneBase
from detr.models.backbones.utils import download_weights


def get_device():
    """Determine the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class PretrainedViT(BackboneBase):
    """Vision Transformer using d4r pretrained weights"""

    def __init__(self, args):
        super().__init__(args)
        self._num_channels = 768
        self._model = self._create_vit()
        self.device = get_device()

        if args.lr_backbone <= 0:
            self.freeze()

        self._load_weights()

    def _create_vit(self):
        """Creates and configures the Vision Transformer model"""
        from timm.models.vision_transformer import VisionTransformer

        model = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
        )
        delattr(model, "head")  # Remove classification head
        return model

    def _load_weights(self):
        """Load pretrained weights"""
        path = download_weights("vit")
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            state_dict = state_dict.get("model", state_dict)
            self._model.load_state_dict(state_dict, strict=False)

    def get_features(self, x):
        """Extract features from input images"""
        x = self._model.patch_embed(x)

        cls_token = self._model.cls_token + self._model.pos_embed[:, :1]
        x = x + self._model.pos_embed[:, 1:]
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self._model.blocks(x)
        x = self._model.norm(x)

        x = x[:, 1:]  # Remove cls token
        N, L, C = x.shape
        H = W = int(L**0.5)
        return x.reshape(N, H, W, C).permute(0, 3, 1, 2)


class PretrainedResNet18(BackboneBase):
    """ResNet18 using d4r pretrained weights"""

    def __init__(self, args):
        super().__init__(args)
        self._num_channels = 512
        self._model = IntermediateLayerGetter(
            self._create_resnet(), return_layers={"layer4": "layer4"}
        )
        self.device = get_device()

        if args.lr_backbone <= 0:
            self.freeze()

        self._load_weights()

    def _create_resnet(self):
        """Creates and configures ResNet18 model"""
        model = models.resnet18(weights=None)
        model.fc = Identity()  # Remove classifier

        def replace_bn_with_gn(module):
            """Replace BatchNorm layers with GroupNorm"""
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    setattr(module, name, nn.GroupNorm(16, child.num_features))
                else:
                    replace_bn_with_gn(child)

        replace_bn_with_gn(model)
        return model

    def _load_weights(self):
        """Load pretrained weights"""
        path = download_weights("resnet18")
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            state_dict = state_dict.get("model", state_dict)
            self._model.load_state_dict(state_dict, strict=False)

    def get_features(self, x):
        return self._model(x)["layer4"]
