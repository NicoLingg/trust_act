from detr.models.backbones.base import BackboneBase
from detr.models.backbones.d4r_pretrained import PretrainedViT, PretrainedResNet18
from detr.models.backbones.huggingface import HFVisionBackbone
from detr.models.backbones.detr_resnet import DetrResnetBackbone

def build_backbone(args) -> BackboneBase:
    """Build backbone based on args"""
    backbone_type = args.backbone.lower()
    
    if backbone_type == 'd4r_vit':
        return PretrainedViT(args)
    elif backbone_type == 'd4r_resnet18':
        return PretrainedResNet18(args)
    elif backbone_type == 'hf_vit':
        return HFVisionBackbone(args)
    elif backbone_type in ['resnet18', 'resnet34', 'resnet50']:
        return DetrResnetBackbone(args)
    else:
        raise ValueError(f'Unknown backbone type: {backbone_type}')

__all__ = ['build_backbone']




