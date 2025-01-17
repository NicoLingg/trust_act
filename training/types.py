from enum import Enum


class TrustRewardSetting(str, Enum):
    # Used during inference
    NONE = "NONE"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class TrustModelling(str, Enum):
    NONE = "NONE"
    # Reward model, scores action sequences based on state
    REWARD = "REWARD"
    # Add trust as contextual input to action transformer
    CONTEXT = "CONTEXT"
    # Use trust as input and predict trust reward
    BOTH = "BOTH"


class TrustRewardReduction(str, Enum):
    MEAN = "MEAN"
    FIRST = "FIRST"
    LAST = "LAST"


AUXILLARY_TASK_DIMS = {
    "joint_velocities": 7,
    "ee_pos_quats": 6,
    "ee_vels": 6,
    "ee_forces": 6,
}

IMAGE = "image"
ROBOT_STATE = "robot_state"  # formerly qpos or joint positions
TRUST = "trust"
ACTIONS = "actions"
AUX_TARGETS = "aux_targets"

# NB: ordering needs to match forward of policy & cvae
STANDARD_INPUTS = ROBOT_STATE, IMAGE, ACTIONS, TRUST
