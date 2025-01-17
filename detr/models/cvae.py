"""
CVAEModel & Outputs class
"""

import dataclasses
import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor, nn
from typing import NamedTuple, Optional
from .backbones import build_backbone
from .position_encoding import build_position_encoding

from .transformer import (
    build_transformer,
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from training.types import (
    TrustModelling,
    TrustRewardSetting,
    TrustRewardReduction,
    AUXILLARY_TASK_DIMS,
)


@dataclasses.dataclass
class CVAEOutputs:
    actions: Tensor
    actions_sample: Optional[Tensor] = None
    latent_mu: Optional[Tensor] = None
    latent_logvar: Optional[Tensor] = None
    latent_sample: Optional[Tensor] = None
    trust_reward: Optional[Tensor] = None
    trust_reward_sample: Optional[Tensor] = None
    aux_losses: Optional[dict] = None

    def for_inference(self):
        return (
            self.actions,
            self.trust_reward,
            self.latent_sample,
            self.actions_sample,
            self.trust_reward_sample,
        )


class InferenceSettings(NamedTuple):
    trust_context_value: float = 1
    num_samples: int = 10
    sampling_stdev: float = 0.01
    trust_reward_setting: TrustRewardSetting = TrustRewardSetting.HIGH
    trust_reward_reduction: TrustRewardReduction = TrustRewardReduction.MEAN

    def __post_init__(self):
        if not (0 <= self.trust_context_value <= 1):
            raise ValueError(
                f"trust_context_value must be between 0 and 1, got {self.trust_context_value}."
            )
        if self.sampling_stdev < 0:
            raise ValueError(
                f"sampling_stdev must be non-negative, got {self.sampling_stdev}."
            )


TRUST_SETTING_TO_VALUE = {"HIGH": 1, "MEDIUM": 0.5, "LOW": 0}


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = torch.randn_like(std)
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def build_encoder(args):
    d_model = args.hidden_dim
    dropout = args.dropout
    nhead = args.nheads
    dim_feedforward = args.dim_feedforward
    num_encoder_layers = args.enc_layers  # TODO shared with VAE decoder
    normalize_before = args.pre_norm
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(
        d_model, nhead, dim_feedforward, dropout, activation, normalize_before
    )
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


class CVAE(nn.Module):
    """CVAE model for trust-aware ACT policy"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_encoder = getattr(args, "use_encoder", True)

        # Core model parameters
        self.M = args.num_image_tokens_sqrt
        self.N = args.num_hist_steps
        self.camera_names = list(args.camera_names)
        hidden_dim = args.hidden_dim
        self.mask_p = args.mask_p
        self.condition_encoder_on_images = args.condition_encoder_on_images

        # Vision backbone
        backbone = build_backbone(args)
        self.backbones = nn.ModuleList([backbone])
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.pos_emb_vision_features = build_position_encoding(args)

        # Main transformer (encder & decoder) and basic components
        self.transformer = build_transformer(args)
        self.input_proj_robot_state = nn.Linear(args.robot_state_dim, hidden_dim)
        self.action_steps_pos_embed = nn.Embedding(args.num_action_steps, hidden_dim)
        self.robot_pos_embed = nn.Embedding(1, hidden_dim)
        self.trust_pos_embed = nn.Embedding(1, hidden_dim)

        # Trust embedding parameters
        self.trust_embedding_dim = hidden_dim
        self.num_trust_levels = 3  # Three discrete trust levels
        self.trust_level_bounds = [0.33, 0.66]  # Boundaries between trust levels

        # Embedding for categorical trust inputs in transformer
        self.decoder_trust_embeddings = nn.Embedding(
            num_embeddings=self.num_trust_levels, embedding_dim=self.trust_embedding_dim
        )

        # Action heads
        self.action_head = nn.Linear(hidden_dim, args.action_dim, bias=True)

        # Auxiliary action loss heads
        self.action_aux_loss_heads = None
        if args.action_aux_loss is not None:
            heads = dict()
            for key in args.action_aux_loss:
                heads[key] = nn.Linear(hidden_dim, AUXILLARY_TASK_DIMS[key])
            self.action_aux_loss_heads = torch.nn.ModuleDict(heads)

        # Trust components
        self.trust_prediction_encoder = None
        self.trust_prediction_head = None
        self.trust_context_condition = False

        # Setup trust reward prediction if enabled
        if args.trust_modelling.value in [
            TrustModelling.REWARD.value,
            TrustModelling.BOTH.value,
        ]:
            self.trust_prediction_encoder = build_encoder(args)
            # build pos embedding for trust prediction encoder cls + actions + vision (vision will be concat later)
            pos_table_trust = get_sinusoid_encoding_table(
                1 + args.num_action_steps, hidden_dim
            )
            pos_table_trust.requires_grad = False
            pos_table_trust = pos_table_trust.permute(1, 0, 2)
            self.register_buffer("encoder_trust_pos_embed", pos_table_trust)
            self.trust_prediction_head = nn.Linear(hidden_dim, 1)
            # Action projection layer is not shared between encoder and trust prediction encoder
            self.trust_prediction_encoder_action_proj = nn.Linear(
                args.action_dim, hidden_dim
            )
            # Ensures that gradients from trust prediction encoder do not flow to vision backbone
            self.trust_prediction_stop_gradient = args.trust_prediction_stop_gradient

        # CVAE encoder components - only create if using encoder
        if self.use_encoder:
            # Setup trust context conditioning in encoder if enabled
            if args.trust_modelling.value in [
                TrustModelling.CONTEXT.value,
                TrustModelling.BOTH.value,
            ]:
                self.trust_context_condition = True
                # Embedding for categorical trust inputs in encoder
                self.encoder_trust_embeddings = nn.Embedding(
                    num_embeddings=self.num_trust_levels,
                    embedding_dim=self.trust_embedding_dim,
                )

            self.latent_dim = args.latent_dim
            self.encoder = build_encoder(args)
            self.cls_pos_embed = nn.Embedding(1, hidden_dim)
            self.encoder_robot_state_proj = nn.Linear(args.robot_state_dim, hidden_dim)
            self.encoder_action_proj = nn.Linear(args.action_dim, hidden_dim)
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
            self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
            encoder_extra_tokens = 2  # cls and robot_state
            if self.trust_context_condition:
                encoder_extra_tokens += 1

            # Encoder positional embeddings
            pos_table = get_sinusoid_encoding_table(
                encoder_extra_tokens + args.num_action_steps, hidden_dim
            )
            pos_table.requires_grad = False
            pos_table = pos_table.permute(1, 0, 2)
            self.register_buffer("encoder_pos_embed", pos_table)

            # Postion embedding for latent in decoder's encoder
            self.latent_pos_embed = nn.Embedding(1, hidden_dim)

        # Multi-timestep positional embeddings
        if self.N > 1:
            time_pos_table = get_sinusoid_encoding_table(self.N, hidden_dim)
            time_pos_table.requires_grad = False
            token_per_t = (
                self.M * self.M * len(self.camera_names) + 1
            )  # vision tokens + robot state

            if self.use_encoder:
                repeat_pattern = torch.tensor(
                    [token_per_t] * (self.N - 1) + [token_per_t + 1]
                )
            else:
                repeat_pattern = torch.tensor([token_per_t] * self.N)

            time_pos_embed = torch.repeat_interleave(
                time_pos_table, repeat_pattern, dim=1
            )
            self.register_buffer("time_pos_embed", time_pos_embed)

        # Print parameter counts
        backbone_params = sum(
            p.numel()
            for name, p in self.named_parameters()
            if p.requires_grad and "backbones" in name
        )
        network_params = sum(
            p.numel()
            for name, p in self.named_parameters()
            if p.requires_grad and "backbones" not in name
        )
        total_params = backbone_params + network_params

        print(f"⚙ Model parameters:")
        print(f"  • Backbone: {backbone_params/1e6:.2f}M")
        print(f"  • Network: {network_params/1e6:.2f}M")
        print(f"  → Total: {total_params/1e6:.2f}M")

    def get_trust_level(self, trust_value):
        if trust_value.dim() == 2:
            trust_value = trust_value.squeeze(-1)

        # Clip trust values to [0, 1]
        trust_value = torch.clamp(trust_value, 0.0, 1.0)

        # Convert to trust levels
        trust_level = torch.zeros_like(trust_value, dtype=torch.long)
        trust_level[trust_value > self.trust_level_bounds[0]] = 1
        trust_level[trust_value > self.trust_level_bounds[1]] = 2

        return trust_level

    def get_trust_embedding(self, trust_value, mode="encoder"):
        """Get trust embedding for either encoder or decoder
        trust_value: tensor of shape (bs,) or (bs, seq)
        mode: either 'encoder' or 'decoder'
        """
        trust_level = self.get_trust_level(trust_value)
        if mode == "encoder":
            return self.encoder_trust_embeddings(trust_level)
        else:
            return self.decoder_trust_embeddings(trust_level)

    def forward(
        self,
        robot_state: Tensor,
        image: Tensor,
        actions: Optional[Tensor] = None,
        trust: Optional[Tensor] = None,
        inference_settings: Optional[InferenceSettings] = None,
    ):
        """
        robot_state: batch, num_hist_steps (N), robot_state_dim
        image: batch, num_hist_steps (N), num_cam, channel, height, width
        actions: batch, num_action_steps, action_dim
        inference_settings
        """
        is_training = actions is not None  # train or val
        bs = robot_state.shape[0]
        latent_input = None
        vision_input, vision_pos_embed = self._forward_backbone(image)

        if self.use_encoder:
            if is_training:
                assert inference_settings is None

                if self.condition_encoder_on_images:
                    latent_input, mu, logvar = self._forward_encoder(
                        robot_state=robot_state[:, -1],
                        actions=actions,
                        trust=trust,
                        vision_input=vision_input,
                        vision_pos_embed=vision_pos_embed,
                    )
                else:
                    latent_input, mu, logvar = self._forward_encoder(
                        robot_state=robot_state[:, -1], actions=actions, trust=trust
                    )
            else:
                assert inference_settings is not None
                assert bs == 1, "Single batch inference supported"

                if inference_settings.num_samples is not None:
                    bs = inference_settings.num_samples

                latent_sample = torch.zeros(
                    [bs, self.latent_dim], dtype=torch.float32
                ).to(robot_state.device)

                if inference_settings.sampling_stdev > 0:
                    latent_sample = latent_sample.normal_(
                        std=inference_settings.sampling_stdev
                    )

                latent_input = self.latent_out_proj(latent_sample)

        robot_state_input = self.input_proj_robot_state(robot_state)
        trust_input = self.get_trust_embedding(torch.mean(trust, dim=1), mode="decoder")

        inputs = self._prepare_inputs(
            vision_input, robot_state_input, latent_input, trust_input, is_training, bs
        )
        input_pos_embeds = self._prepare_input_pos_embeds(vision_pos_embed, bs)
        # Repeat once per sample: # 60, bs, 512
        action_steps_pos_embed = self.action_steps_pos_embed.weight.unsqueeze(1).repeat(
            1, bs, 1
        )

        # Initialise action query
        action_query = torch.zeros_like(action_steps_pos_embed)

        # hidden_state: batch_size / num_samples, seq, hidden_dim
        hidden_state, _ = self.transformer(
            action_query=action_query,
            inputs=inputs,
            inputs_pos_embed=input_pos_embeds,
            action_steps_pos_embed=action_steps_pos_embed,
        )

        predicted_actions = self.action_head(hidden_state)
        outputs = CVAEOutputs(actions=predicted_actions)

        if is_training:
            if self.trust_prediction_encoder:
                predicted_trust = self._forward_trust_encoder(
                    actions=actions,
                    vision_input=vision_input,
                    vision_pos_embed=vision_pos_embed,
                )
                outputs.trust_reward = predicted_trust

            if self.use_encoder:
                outputs.latent_mu = mu
                outputs.latent_logvar = logvar

            if self.action_aux_loss_heads:
                aux_losses = dict()
                for key, head in self.action_aux_loss_heads.items():
                    aux_losses[key] = head(hidden_state)
                outputs.aux_losses = aux_losses

        else:  # Inference mode
            # Use predicted actions during inference
            if self.trust_prediction_encoder:
                vision_input = vision_input.repeat(inference_settings.num_samples, 1, 1, 1)
                predicted_trust = self._forward_trust_encoder(
                    actions=predicted_actions,
                    vision_input=vision_input,
                    vision_pos_embed=vision_pos_embed,
                )
                outputs.trust_reward = predicted_trust

            # Logic for search over repeated samples
            if bs > 1:
                actions_sample = predicted_actions.clone().detach()
                best_idx, trust_reward_sample = self._best_of_n_search(
                    inference_settings, predicted_trust, bs
                )
                predicted_actions = actions_sample[best_idx]
                predicted_actions = predicted_actions.unsqueeze(0)
                predicted_trust = predicted_trust[best_idx]
                predicted_trust = predicted_trust.unsqueeze(0)

                outputs = CVAEOutputs(
                    actions=predicted_actions,
                    actions_sample=actions_sample,
                    latent_sample=latent_sample,
                    trust_reward=predicted_trust,
                    trust_reward_sample=trust_reward_sample,
                )

        return outputs

    def _forward_trust_encoder(
        self,
        actions: Tensor,
        vision_input: Tensor = None,
        vision_pos_embed: Tensor = None,
    ):
        if self.trust_prediction_stop_gradient:
            vision_input = vision_input.detach()
            vision_pos_embed = vision_pos_embed.detach()

        bs = actions.shape[0]
        actions = self.trust_prediction_encoder_action_proj(
            actions
        )  # (bs, seq, hidden_dim)

        cls_embed = self.cls_pos_embed.weight  # (1, hidden_dim)
        cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(
            bs, 1, 1
        )  # (bs, 1, hidden_dim)

        vision_pos_embed = vision_pos_embed.flatten(
            start_dim=1, end_dim=2
        )  # (action_steps + 1 + 1, 1, bs)
        vision_pos_embed = vision_pos_embed.transpose(
            0, 1
        )  # (1, action_steps + 1 + 1, bs)

        encoder_pos_embed = torch.cat(
            [self.encoder_trust_pos_embed, vision_pos_embed], dim=0
        )

        vision_input = vision_input.flatten(start_dim=1, end_dim=2)
        encoder_input = torch.cat(
            [cls_embed, actions, vision_input], axis=1
        )  # (bs, seq+1+1, hidden_dim)

        encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1+1, bs, hidden_dim)

        # query model
        trust_out = self.trust_prediction_encoder(
            inputs=encoder_input, inputs_pos_embed=encoder_pos_embed
        )

        trust = self.trust_prediction_head(trust_out[0])  # take cls output only
        trust = torch.sigmoid(trust)

        return trust

    def _forward_encoder(
        self,
        robot_state: Tensor,
        actions: Tensor,
        trust: Optional[Tensor] = None,
        vision_input: Optional[Tensor] = None,
        vision_pos_embed: Optional[Tensor] = None,
    ):
        # project action sequence to embedding dim, and concat with a CLS token
        bs = robot_state.shape[0]
        actions = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
        trust_embed = None

        # Only add trust to actions if trust_in_cvae_encoder is enabled
        if self.trust_context_condition:
            assert (
                trust is not None
            ), "Trust values required when condition_encoder_on_trust is enabled"
            # trust comes in as trust.shape = (bs, seq, 1)
            trust = trust.squeeze(-1)  # trust.shape = (bs, seq)
            # take mean over sequence dimension
            trust = torch.mean(trust, dim=1)  # trust.shape = (bs,)
            trust_embed = self.get_trust_embedding(
                trust, mode="encoder"
            )  # trust_embed.shape = (bs, hidden_dim)
            trust_embed = torch.unsqueeze(
                trust_embed, axis=1
            )  # trust_embed.shape = (bs, 1, hidden_dim)

        robot_state = self.encoder_robot_state_proj(robot_state)  # (bs, hidden_dim)
        robot_state = torch.unsqueeze(robot_state, axis=1)  # (bs, 1, hidden_dim)

        cls_embed = self.cls_pos_embed.weight  # (1, hidden_dim)
        cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(
            bs, 1, 1
        )  # (bs, 1, hidden_dim)

        if self.condition_encoder_on_images:
            assert vision_input is not None
            vision_pos_embed = vision_pos_embed.flatten(
                start_dim=1, end_dim=2
            )  # (action_steps + 1 + 1, 1, bs)
            vision_pos_embed = vision_pos_embed.transpose(
                0, 1
            )  # (1, action_steps + 1 + 1, bs)

            encoder_pos_embed = torch.cat(
                [self.encoder_pos_embed, vision_pos_embed], dim=0
            )

            vision_input = vision_input.flatten(start_dim=1, end_dim=2)
            if trust_embed is not None:
                encoder_input = torch.cat(
                    [cls_embed, robot_state, trust_embed, actions, vision_input], axis=1
                )  # (bs, seq+1+1+1, hidden_dim)
            else:
                encoder_input = torch.cat(
                    [cls_embed, robot_state, actions, vision_input], axis=1
                )  # (bs, seq+1+1, hidden_dim)

        else:
            encoder_pos_embed = self.encoder_pos_embed
            if trust_embed is not None:
                encoder_input = torch.cat(
                    [cls_embed, robot_state, trust_embed, actions], axis=1
                )  # (bs, seq+1+1+1, hidden_dim)
            else:
                encoder_input = torch.cat(
                    [cls_embed, robot_state, actions], axis=1
                )  # (bs, seq+1+1, hidden_dim)
        encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1+1, bs, hidden_dim)

        # query model
        encoder_output = self.encoder(
            inputs=encoder_input, inputs_pos_embed=encoder_pos_embed
        )
        # CAVE latent style variable
        latent_info = self.latent_proj(encoder_output[0])  # take cls output only
        mu = latent_info[:, : self.latent_dim]
        logvar = latent_info[:, self.latent_dim :]
        latent_sample = reparametrize(mu, logvar)
        latent_input = self.latent_out_proj(latent_sample)

        return latent_input, mu, logvar

    def _forward_backbone(self, image: Tensor):
        # Image observation features and position embeddings
        # image is shape: bs, seq, cams, C, H, W
        bs = image.shape[0]
        # assert cams == len(self.camera_names)
        all_cam_features = []
        all_cam_pos = []
        for cam_id, cam_name in enumerate(self.camera_names):
            cam_images = image[:, :, cam_id]
            # Flatten history dim into batch dimension
            cam_images = cam_images.flatten(start_dim=0, end_dim=1)
            features, _ = self.backbones[0](cam_images)
            # print(f'Features shape before adaptive_avg_pool: {features.shape}')

            features = F.adaptive_avg_pool2d(features, output_size=(self.M, self.M))
            features = self.input_proj(features)
            pos = self.pos_emb_vision_features(features).unsqueeze(0)

            # Unflatten batch dimension
            features = torch.unflatten(features, dim=0, sizes=(bs, self.N))

            all_cam_features.append(features)
            all_cam_pos.append(pos)

        # bs, N, D, M, M, NC
        vision_input = torch.stack(all_cam_features, axis=-1)
        # bs, N, D, M*M*NC
        vision_input = vision_input.flatten(start_dim=3, end_dim=5)
        # transpose hidden and spatial dimensions: bs, N, M*M*NC, D
        vision_input = vision_input.transpose(2, 3)

        # 1, 1, D, M, M, NC
        vision_pos_embed = torch.stack(all_cam_pos, axis=-1)
        vision_pos_embed = vision_pos_embed.flatten(start_dim=3, end_dim=5)
        # 1, 1, M*M*NC, D
        vision_pos_embed = vision_pos_embed.transpose(2, 3)

        return vision_input, vision_pos_embed

    def _prepare_inputs(
        self,
        vision_input,
        robot_state_input,
        latent_input,
        trust_input,
        is_training,
        batch_size,
    ):

        if not is_training:
            # Repeat once per latent sample
            vision_input = vision_input.repeat(batch_size, 1, 1, 1)
            robot_state_input = robot_state_input.repeat(batch_size, 1, 1)
            trust_input = trust_input.repeat(batch_size, 1)

        # bs, N, 1, D
        robot_state_input = robot_state_input.unsqueeze(2)

        trust_input = trust_input.unsqueeze(1).unsqueeze(2)
        # concatenate along spatial dimension
        tokens_per_timestep = torch.cat(
            [vision_input, robot_state_input, trust_input], dim=2
        )
        # flatten time & spatial dimensions
        inputs = tokens_per_timestep.flatten(start_dim=1, end_dim=2)

        # Partial masking logic to prevent posterior collapse
        if self.mask_p > 0:
            assert self.use_encoder
            mask = torch.randn(batch_size) < self.mask_p
            inputs[mask] = 0

        # Only append latent if we use it
        if self.use_encoder:
            inputs = torch.cat([inputs, latent_input.unsqueeze(1)], axis=1)

        # transformer wants time, bs, D
        inputs = inputs.transpose(0, 1)
        return inputs

    def _prepare_input_pos_embeds(self, vision_pos_embed, bs):
        # 1, 1, 1, D
        robot_pos_embed = self.robot_pos_embed.weight.unsqueeze(0).unsqueeze(0)
        trust_pos_embed = self.trust_pos_embed.weight.unsqueeze(0).unsqueeze(0)

        # 1, 1, M*M*2+1, D
        pos_embeds_per_timestep = torch.cat(
            [vision_pos_embed, robot_pos_embed, trust_pos_embed], dim=2
        )
        input_pos_embeds = pos_embeds_per_timestep.repeat(bs, self.N, 1, 1).flatten(
            start_dim=1, end_dim=2
        )

        # Only add latent position embedding if we're using encoder/latents
        if self.use_encoder:
            latent_pos_embed = self.latent_pos_embed.weight.unsqueeze(0).repeat(
                bs, 1, 1
            )
            input_pos_embeds = torch.cat([input_pos_embeds, latent_pos_embed], axis=1)

        if self.N > 1:
            time_pos_embeds = self.time_pos_embed.repeat(bs, 1, 1)
            input_pos_embeds += time_pos_embeds
        # transformer wants time, bs, D
        input_pos_embeds = input_pos_embeds.transpose(0, 1)
        return input_pos_embeds

    def _best_of_n_search(
        self,
        inference_settings: InferenceSettings,
        trust_reward: Tensor,
        batch_size: int,
    ):
        if inference_settings.trust_reward_reduction == TrustRewardReduction.MEAN.value:
            trust_reward_samples = torch.mean(trust_reward, dim=1)
        elif (
            inference_settings.trust_reward_reduction
            == TrustRewardReduction.FIRST.value
        ):
            trust_reward_samples = trust_reward[:, 0]
        elif (
            inference_settings.trust_reward_reduction == TrustRewardReduction.LAST.value
        ):
            trust_reward_samples = trust_reward[:, -1]
        else:
            raise NotImplementedError()

        _, indices = torch.sort(trust_reward_samples, dim=0, descending=True)

        if inference_settings.trust_reward_setting == TrustRewardSetting.HIGH.value:
            best_idx = indices[0]
        elif inference_settings.trust_reward_setting == TrustRewardSetting.MEDIUM.value:
            best_idx = indices[batch_size // 2]
        elif inference_settings.trust_reward_setting == TrustRewardSetting.LOW.value:
            best_idx = indices[-1]
        else:
            raise NotImplementedError()

        return best_idx, trust_reward_samples
