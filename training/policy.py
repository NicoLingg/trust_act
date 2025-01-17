import torch
import torch.nn as nn
from torch.nn import functional as F

from training.losses import kl_divergence
from detr.main import build_ACT_model_and_optimizer
from detr.models.cvae import CVAEOutputs, InferenceSettings
from training.types import AUXILLARY_TASK_DIMS, TrustModelling


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE encoder & decoder
        self.optimizer = optimizer
        self.kl_scheduler = (
            None  # Will be set later - if not it will use default_kl_weight
        )
        self.default_kl_weight = args_override.train.kl_scheduler["default_kl_weight"]
        self.action_loss_weight = args_override.policy["action_loss_weight"]

        # Normal training case
        self.action_aux_loss = dict(args_override.policy["action_aux_loss"])
        for loss_name, loss_weight in self.action_aux_loss.items():
            assert loss_name in list(AUXILLARY_TASK_DIMS.keys()) and loss_weight >= 0

        self.trust_reward_loss_weight = 0
        if args_override.policy["trust_modelling"].value in [
            TrustModelling.REWARD.value,
            TrustModelling.BOTH.value,
        ]:
            self.trust_reward_loss_weight = args_override.policy[
                "trust_reward_loss_weight"
            ]

        self.inference_settings = InferenceSettings(
            trust_context_value=args_override.inference["trust_context_value"],
            num_samples=args_override.inference["num_samples"],
            sampling_stdev=args_override.inference["sampling_stdev"],
            trust_reward_setting=args_override.inference["trust_reward_setting"],
            trust_reward_reduction=args_override.inference["trust_reward_reduction"],
        )

    def set_kl_scheduler(self, scheduler):
        """Sets KL scheduler instance created in trainer.py"""
        self.kl_scheduler = scheduler

    def get_kl_weight(self):
        """Helper function to get the KL loss weight."""
        if self.kl_scheduler is not None:
            return self.kl_scheduler.get_kl_loss_weight()
        return self.default_kl_weight

    def forward(self, robot_state, image, actions=None, trust=None, aux_targets=None):
        if actions is not None:  # training time

            loss_dict = {"loss": 0.0}  # Initialize total loss to 0
            latent_dict = dict()

            outputs: CVAEOutputs = self.model(
                robot_state=robot_state, image=image, actions=actions, trust=trust
            )

            latent_dict["mu"] = outputs.latent_mu
            latent_dict["logvar"] = outputs.latent_logvar

            if self.model.use_encoder:
                total_kld, _, _ = kl_divergence(outputs)
                kl_weight = self.get_kl_weight()
                loss_dict["kl"] = total_kld[0].detach().clone() * kl_weight
                loss_dict["loss"] += total_kld[0] * kl_weight

            a_hat = outputs.actions
            l1 = F.l1_loss(input=a_hat, target=actions, reduction="mean")
            loss_dict["l1"] = l1.detach().clone() * self.action_loss_weight
            loss_dict["loss"] += l1 * self.action_loss_weight

            # During training
            if self.trust_reward_loss_weight > 0:
                trust_reward = outputs.trust_reward
                trust_reward_loss = F.mse_loss(
                    trust_reward, target=trust.mean(dim=1), reduction="mean"
                )
                # For logging/reporting, also calculate MAE
                trust_reward_mae = F.l1_loss(
                    trust_reward, target=trust.mean(dim=1), reduction="mean"
                )
                loss_dict["trust_reward_mae"] = (
                    trust_reward_mae.detach().clone()
                )  # for interpretation
                loss_dict["trust_reward"] = (
                    trust_reward_loss.detach().clone() * self.trust_reward_loss_weight
                )
                loss_dict["loss"] += trust_reward_loss * self.trust_reward_loss_weight

            if len(self.action_aux_loss) > 0:
                assert outputs.aux_losses is not None
                for name, weight in self.action_aux_loss.items():
                    loss = F.mse_loss(
                        outputs.aux_losses[name], aux_targets[name], reduction="mean"
                    )
                    loss_dict[name] = loss.detach().clone() * weight
                    loss_dict["loss"] += loss * weight

            return loss_dict, latent_dict

        else:  # inference time
            outputs: CVAEOutputs = self.model(
                robot_state=robot_state,
                image=image,
                trust=trust,
                inference_settings=self.inference_settings,
            )  # no action, sample from prior
            return outputs.for_inference()

    def _load_policy_weights(self, checkpoint_path):

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        new_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                new_key = key.replace("_orig_mod.", "")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        self.load_state_dict(new_state_dict)
