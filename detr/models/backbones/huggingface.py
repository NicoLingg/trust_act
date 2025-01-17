from transformers import AutoModel
from detr.models.backbones.base import BackboneBase


class HFVisionBackbone(BackboneBase):
    """HuggingFace Vision Models"""

    def __init__(self, args):
        super().__init__(args)
        model_name = getattr(args, "hf_model_name", "google/vit-base-patch16-224")
        self._model = AutoModel.from_pretrained(model_name)
        self._num_channels = self._model.config.hidden_size

        if not args.lr_backbone > 0:
            self.freeze()

    def get_features(self, x):
        outputs = self._model(x, output_hidden_states=True)
        features = outputs.last_hidden_state

        # Convert [B, L, C] -> [B, C, H, W]
        if len(features.shape) == 3:
            B, L, C = features.shape
            H = W = int((L - 1) ** 0.5)  # -1 for cls token
            features = features[:, 1:].reshape(B, H, W, C)
            features = features.permute(0, 3, 1, 2)

        return features
