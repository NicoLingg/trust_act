import torch
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter
from detr.models.backbones.base import BackboneBase


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class DetrResnetBackbone(BackboneBase):
    """Original torchvision ResNet backbone"""

    def __init__(self, args):
        super().__init__(args)
        backbone_name = args.backbone  # e.g., 'resnet18'

        backbone = getattr(models, backbone_name)(
            weights="DEFAULT", norm_layer=FrozenBatchNorm2d
        )

        if not args.lr_backbone > 0:
            self.freeze()

        return_layers = {"layer4": "layer4"}
        self._model = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self._num_channels = 512 if backbone_name in ("resnet18", "resnet34") else 2048

    def get_features(self, x):
        return self._model(x)["layer4"]
