import os
import logging
import hydra
import pybullet as p
import pybullet_data
import time
import numpy as np
import torch
import yaml
import cv2
from omegaconf import DictConfig
from enum import Enum
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import torchvision.io as io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.policy import ACTPolicy
from training.types import TrustModelling, TrustRewardSetting, TrustRewardReduction
from training.utils import set_seed
from detr.models.cvae import CVAEOutputs, InferenceSettings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualizationMode(Enum):
    RECORDED = "recorded"
    POLICY_FROM_TIMESTAMP = "policy_from_timestamp"


class RobotSimulation:
    def __init__(
        self,
        urdf_path: str,
        stats_path: str,
        end_effector_link_name: str = "gripper_tool0",
    ):
        """Initialize robot simulation environment."""
        self.end_effector_link_name = end_effector_link_name
        self.robotId, self.joint_info = self._initialize_simulation(urdf_path)
        self.stats = self._load_stats(stats_path)
        self.gripper_joints = self._get_gripper_joints()
        logger.info(f"Initialized robot with {len(self.joint_info)} joints")
        logger.info(f"Found {len(self.gripper_joints)} gripper joints")

    @staticmethod
    def _load_stats(stats_path: str) -> dict:
        """Load statistics for normalization."""
        try:
            with open(stats_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading stats from {stats_path}: {e}")
            raise

    def _get_gripper_joints(self) -> list:
        """Get all gripper-related joints."""
        gripper_joints = []

        # Define the joint names we want to control
        controlled_joints = [
            "gripper_finger_1_joint_1",
            "gripper_finger_1_joint_2",
            "gripper_finger_1_joint_3",
            "gripper_finger_2_joint_1",
            "gripper_finger_2_joint_2",
            "gripper_finger_2_joint_3",
            "gripper_finger_middle_joint_1",
            "gripper_finger_middle_joint_2",
            "gripper_finger_middle_joint_3",
        ]

        num_joints = p.getNumJoints(self.robotId)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robotId, i)
            joint_name = joint_info[1].decode("utf-8")

            if joint_name in controlled_joints:
                gripper_joints.append(
                    {
                        "jointIndex": joint_info[0],
                        "jointName": joint_name,
                        "lowerLimit": joint_info[8],
                        "upperLimit": joint_info[9],
                        "maxForce": joint_info[10],
                        "maxVelocity": joint_info[11],
                    }
                )

        return gripper_joints

    def _initialize_simulation(self, urdf_path: str):
        """Initialize PyBullet simulation and load robot."""
        physicsClient = p.connect(p.GUI)

        # Disable GUI elements
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        robotStartPos = [0, 0, 0.2]
        robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        try:
            robotId = p.loadURDF(
                fileName=urdf_path,
                basePosition=robotStartPos,
                baseOrientation=robotStartOrientation,
                useFixedBase=True,
                flags=p.URDF_USE_SELF_COLLISION,
            )
        except Exception as e:
            logger.error(f"Error loading URDF from {urdf_path}: {e}")
            raise

        # Set camera position
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=50.0,
            cameraPitch=-35.0,
            cameraTargetPosition=[0.5, 0.0, 0.5],
        )

        joint_info = self._get_joint_info(robotId)
        self._configure_joints(robotId, joint_info)

        p.setRealTimeSimulation(0)
        p.setTimeStep(1 / 30.0)

        return robotId, joint_info

    def _get_joint_info(self, robotId) -> list:
        """Get information about all joints in the robot."""
        joint_info = []
        num_joints = p.getNumJoints(robotId)
        logger.info(f"Found {num_joints} joints in total")

        target_joints = [
            "robot_shoulder_pan_joint",
            "robot_shoulder_lift_joint",
            "robot_elbow_joint",
            "robot_wrist_1_joint",
            "robot_wrist_2_joint",
            "robot_wrist_3_joint",
        ]

        for i in range(num_joints):
            info = p.getJointInfo(robotId, i)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            logger.debug(
                f"Joint {i}: {joint_name} (type: {info[2]}) links to {link_name}"
            )

            if joint_name in target_joints:
                joint_info.append(
                    {
                        "jointIndex": info[0],
                        "jointName": joint_name,
                        "jointType": info[2],
                        "lowerLimit": info[8],
                        "upperLimit": info[9],
                        "maxForce": info[10],
                        "maxVelocity": info[11],
                        "targetIndex": target_joints.index(joint_name),
                    }
                )

        joint_info.sort(key=lambda x: x["targetIndex"])
        return joint_info

    def _configure_joints(self, robotId, joint_info):
        """Configure joint parameters for smooth motion."""
        for joint in joint_info:
            # Disable velocity control
            p.setJointMotorControl2(
                robotId, joint["jointIndex"], p.VELOCITY_CONTROL, force=0
            )

            # Configure joint dynamics
            p.changeDynamics(
                robotId,
                joint["jointIndex"],
                jointDamping=0.1,
                linearDamping=0.1,
                angularDamping=0.1,
            )

    def set_gripper_position(self, position: float):
        """
        Set the gripper position (0 = fully open, 1 = fully closed).
        """
        position = np.clip(position, 0, 1)

        # Define joint positions for each finger based on the gripper position
        joint_positions = {
            # Finger 1
            "gripper_finger_1_joint_1": 0.0495 + position * (1.2218 - 0.0495),
            "gripper_finger_1_joint_2": position * 1.5708,
            "gripper_finger_1_joint_3": -0.0523 + position * (-1.2217 + 0.0523),
            # Finger 2
            "gripper_finger_2_joint_1": 0.0495 + position * (1.2218 - 0.0495),
            "gripper_finger_2_joint_2": position * 1.5708,
            "gripper_finger_2_joint_3": -0.0523 + position * (-1.2217 + 0.0523),
            # Middle finger
            "gripper_finger_middle_joint_1": 0.0495 + position * (1.2218 - 0.0495),
            "gripper_finger_middle_joint_2": position * 1.5708,
            "gripper_finger_middle_joint_3": -0.0523 + position * (-1.2217 + 0.0523),
        }

        # Apply positions to all gripper joints
        for joint in self.gripper_joints:
            target_pos = joint_positions[joint["jointName"]]
            p.setJointMotorControl2(
                bodyIndex=self.robotId,
                jointIndex=joint["jointIndex"],
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=joint["maxForce"],
                maxVelocity=joint["maxVelocity"] * 0.5,
                positionGain=0.3,
                velocityGain=1.0,
            )

    def apply_action(self, action, step_time: float = 1 / 30.0):
        """Apply action and step simulation."""
        if not isinstance(action, (np.ndarray, list, tuple)):
            raise TypeError(f"Action must be an array-like object, got {type(action)}.")

        action = np.array(action)
        if action.ndim != 1:
            raise ValueError(f"Action must be a 1D array, got shape {action.shape}.")

        if len(action) != len(self.joint_info) + 1:
            raise ValueError(
                f"Expected action length {len(self.joint_info) + 1}, got {len(action)}."
            )

        # Extract gripper position (assuming it's the last element of the action)
        arm_action = action[:-1]
        gripper_position = action[-1]

        # Apply arm joint positions
        self._apply_joint_positions(arm_action)

        # Apply gripper position
        self.set_gripper_position(gripper_position)

        # Calculate steps needed to match the desired step_time
        num_steps = max(
            int(step_time / p.getPhysicsEngineParameters()["fixedTimeStep"]), 1
        )

        # Disable rendering for faster simulation
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        for _ in range(num_steps):
            p.stepSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def _apply_joint_positions(self, joint_positions):
        """Apply the joint positions to the robot arm (excluding gripper)."""
        if len(self.joint_info) != len(joint_positions):
            raise ValueError(
                f"Number of joint positions ({len(joint_positions)}) "
                f"does not match number of joints ({len(self.joint_info)})."
            )

        for i, joint in enumerate(self.joint_info):
            position = np.clip(
                joint_positions[i], joint["lowerLimit"], joint["upperLimit"]
            )

            p.setJointMotorControl2(
                bodyIndex=self.robotId,
                jointIndex=joint["jointIndex"],
                controlMode=p.POSITION_CONTROL,
                targetPosition=position,
                force=joint["maxForce"] * 0.8,
                maxVelocity=joint["maxVelocity"] * 0.5,
                positionGain=0.5,
                velocityGain=1.0,
            )

    def reset_to_position(self, target_position, steps: int = 100):
        """Reset robot to a given joint position smoothly."""
        current_positions = np.array(
            [
                p.getJointState(self.robotId, joint["jointIndex"])[0]
                for joint in self.joint_info
            ]
        )

        # Extract arm positions and gripper position
        target_arm_positions = target_position[:-1]
        target_gripper_position = target_position[-1]

        position_difference = np.abs(target_arm_positions - current_positions)
        max_difference = np.max(position_difference)
        num_steps = max(int(max_difference / 0.01), steps)

        # Disable rendering for faster simulation
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        for t in np.linspace(0, 1, num_steps):
            intermediate_positions = current_positions + t * (
                target_arm_positions - current_positions
            )
            self._apply_joint_positions(intermediate_positions)
            self.set_gripper_position(target_gripper_position)
            p.stepSimulation()

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        # Ensure final position is set
        self._apply_joint_positions(target_arm_positions)
        self.set_gripper_position(target_gripper_position)
        p.stepSimulation()

    def get_end_effector_link_index(self) -> int:
        """Get the link index of the end-effector."""
        num_joints = p.getNumJoints(self.robotId)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robotId, i)
            link_name = joint_info[12].decode("utf-8")
            if link_name == self.end_effector_link_name:
                return i
        raise ValueError(
            f"End effector link '{self.end_effector_link_name}' not found."
        )

    def set_joint_positions(self, joint_positions):
        """Set the joint positions directly without simulation."""
        arm_joint_positions = joint_positions[:-1]  # Exclude gripper position
        for i, joint in enumerate(self.joint_info):
            p.resetJointState(
                bodyUniqueId=self.robotId,
                jointIndex=joint["jointIndex"],
                targetValue=arm_joint_positions[i],
            )
        # Set gripper position
        gripper_position = joint_positions[-1]
        self.set_gripper_position(gripper_position)


class DatasetPolicyVisualizer:
    def __init__(
        self, urdf_path: str, stats_path: str, checkpoint_path: str, cfg: DictConfig
    ):
        self.sim = RobotSimulation(urdf_path, stats_path)
        self.device = self._setup_device()
        self.policy = self._setup_policy(cfg, checkpoint_path)
        self.cfg = cfg
        logger.info(f"Using device: {self.device}")

    def _setup_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _setup_policy(self, cfg: DictConfig, checkpoint_path: str):
        cfg.policy.trust_modelling = TrustModelling[cfg.policy.trust_modelling]
        cfg.inference.trust_reward_setting = TrustRewardSetting[
            cfg.inference.trust_reward_setting
        ]
        policy = ACTPolicy(cfg)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            logger.error(f"Error loading checkpoint from {checkpoint_path}: {e}")
            raise

        state_dict = checkpoint["model_state_dict"]
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                new_key = key.replace("_orig_mod.", "")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        policy.load_state_dict(new_state_dict)
        policy.to(self.device)
        policy.eval()
        return policy

    def save_trajectories_to_npz(
        self,
        policy_trajectories,
        actual_qpos_sequence=None,
        save_path="trajectories.npz",
    ):
        """
        Save end effector positions from policy trajectories and actual trajectory to NPZ file.

        Args:
            policy_trajectories (list): List of predicted trajectories from policy
            actual_qpos_sequence (np.ndarray, optional): Sequence of actual joint positions
            save_path (str): Path to save the NPZ file
        """
        # Get end effector positions for all policy trajectories
        policy_ee_positions = []
        for trajectory in policy_trajectories:
            positions = self._get_end_effector_positions(trajectory)
            # Convert list of tuples to numpy array
            positions = np.array(positions)
            policy_ee_positions.append(positions)

        # Convert list of arrays to single numpy array
        policy_ee_positions = np.array(policy_ee_positions)

        # Save data dictionary
        save_dict = {
            "policy_trajectories": policy_ee_positions,
            "num_trajectories": len(policy_trajectories),
            "trajectory_length": len(policy_trajectories[0]),
        }

        # If actual trajectory is provided, add it to the save dict
        if actual_qpos_sequence is not None:
            actual_positions = self._get_end_effector_positions_from_qpos_sequence(
                actual_qpos_sequence
            )
            actual_positions = np.array(actual_positions)
            save_dict["actual_trajectory"] = actual_positions

        # Save to NPZ file
        try:
            np.savez(save_path, **save_dict)
            logger.info(f"Successfully saved trajectories to {save_path}")

            # Log some statistics
            logger.info(f"Saved data shapes:")
            logger.info(f"- Policy trajectories: {policy_ee_positions.shape}")
            if actual_qpos_sequence is not None:
                logger.info(f"- Actual trajectory: {actual_positions.shape}")

        except Exception as e:
            logger.error(f"Error saving trajectories to {save_path}: {e}")
            raise

    def unnormalize_qpos(self, qpos_normalized: np.ndarray) -> np.ndarray:
        """Unnormalize joint positions using stats."""
        mean = np.array(self.sim.stats["mean"]["observations/joint_positions"])
        std = np.array(self.sim.stats["std"]["observations/joint_positions"])
        qpos = qpos_normalized * std + mean
        return qpos

    def normalize_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Normalize joint positions using stats."""
        mean = np.array(self.sim.stats["mean"]["observations/joint_positions"])
        std = np.array(self.sim.stats["std"]["observations/joint_positions"])
        qpos_normalized = (qpos - mean) / std
        return qpos_normalized

    def load_episode(self, episode_path: str):
        """Load a full episode from an NPZ file with separate image frames directory."""
        try:
            # Load sequence data from NPZ file
            sequence_path = os.path.join(episode_path, "sequence.npz")
            with np.load(sequence_path) as seq_data:
                qpos_sequence = seq_data["joint_positions"]
                actions = seq_data["actions"]
                trust = seq_data["trust_levels"]
                trust = np.clip(trust, 0.0, 1.0)

            # Find all available camera directories
            frames_dir = os.path.join(episode_path, "frames")
            num_frames = len(qpos_sequence)

            # Load images for each frame and camera
            camera_images = {}
            for cam_name in self.cameras:
                images_sequence = []
                for frame_idx in range(num_frames):
                    image_path = os.path.join(
                        frames_dir, f"{frame_idx:06d}", f"{cam_name}.jpg"
                    )
                    image = io.read_image(image_path)

                    if image is None:
                        raise ValueError(f"Failed to load image: {image_path}")

                    image = image / 255.0  # Scale to [0, 1]
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(
                        -1, 1, 1
                    )
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(
                        -1, 1, 1
                    )
                    image = (image - mean) / std
                    images_sequence.append(image)

                camera_images[cam_name] = torch.stack(images_sequence)

            if camera_images:
                camera_images = {k: v.numpy() for k, v in camera_images.items()}
                images = np.stack([camera_images[cam] for cam in self.cameras], axis=1)
                images = np.transpose(images, (0, 1, 3, 4, 2))
            else:
                raise ValueError("No camera images found in the episode data")

            return qpos_sequence, actions, trust, images

        except Exception as e:
            logger.error(f"Error loading episode data from {episode_path}: {e}")
            raise

    def display_image(self, image):
        """Display multiple camera views using OpenCV."""
        if image.ndim == 3:
            image = image[None]

        num_cameras = image.shape[0]

        for cam_idx in range(num_cameras):
            cam_image = image[cam_idx]

            if cam_image.dtype == np.float32 or cam_image.dtype == np.float64:
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                cam_image = (cam_image * std + mean) * 255.0
                cam_image = cam_image.astype(np.uint8)

            image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            cv2.imshow(f"Camera {cam_idx}", image_bgr)

        cv2.waitKey(1)

    def visualize_recorded_trajectory(
        self, qpos_sequence, images, pause_time: float = 1 / 30.0
    ):
        """Visualize a recorded trajectory from the dataset with the ability to replay."""
        logger.info("Visualizing recorded trajectory...")
        logger.debug(f"qpos_sequence shape: {qpos_sequence.shape}")

        exit_simulation = False
        while not exit_simulation:
            initial_position = self.unnormalize_qpos(qpos_sequence[0])
            self.sim.reset_to_position(initial_position)

            actual_positions = self._get_end_effector_positions_from_qpos_sequence(
                qpos_sequence
            )
            self._draw_trajectory(actual_positions, color=[0, 0, 1])

            frame_duration = pause_time
            next_frame_time = time.time() + frame_duration
            for step, qpos_norm in enumerate(
                tqdm(qpos_sequence, desc="Playing recorded trajectory")
            ):
                qpos = self.unnormalize_qpos(qpos_norm)
                self.sim.apply_action(qpos, step_time=pause_time)

                # Display the corresponding image
                if step < len(images):
                    image = images[step]
                    self.display_image(image)

                # Synchronize with real-time
                sleep_time = next_frame_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                next_frame_time += frame_duration

            logger.info("Total playback completed.")

            logger.info("Press 'r' to replay, 'q' to exit...")
            while True:
                keys = p.getKeyboardEvents()
                if ord("r") in keys and keys[ord("r")] & p.KEY_WAS_TRIGGERED:
                    logger.info("Replaying trajectory...")
                    break  # Break the inner loop to replay the trajectory
                if ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED:
                    exit_simulation = True
                    break
                time.sleep(0.1)

        logger.info("Exiting visualization...")

    def generate_policy_trajectories(
        self, initial_qpos_norm, image, actual_trust_sequence
    ):
        """Generate multiple trajectories using the policy starting from a single time step."""
        trajectories = []
        trust_rewards_list = []
        trust_scores = []

        with torch.no_grad():
            curr_qpos_norm = (
                torch.from_numpy(initial_qpos_norm).float().to(self.device).unsqueeze(0)
            )
            image = image[None, None, :, :, :, :]
            image = np.transpose(image, (0, 1, 2, 5, 3, 4))
            curr_image = torch.from_numpy(image).float().to(self.device)

            # genreate new trust action sequence same as the actual trust sequence just all set to 0.5
            actual_trust_sequence = (
                torch.ones_like(actual_trust_sequence)
                * self.cfg.inference.trust_context_value
            )
            print(actual_trust_sequence)

            outputs = self.policy(
                robot_state=curr_qpos_norm,
                image=curr_image,
                actions=None,
                trust=actual_trust_sequence,
                aux_targets=None,
            )

            all_actions, trust_reward, latent_sample, a_hats, trust_scores = outputs

            for i in range(a_hats.shape[0]):
                action_norm_np = a_hats[i].cpu().numpy()
                trajectory = []
                for j in range(action_norm_np.shape[0]):
                    action = self.unnormalize_qpos(action_norm_np[j])
                    trajectory.append(action)
                trajectories.append(trajectory)

            trust_scores = trust_scores.cpu().numpy()

        return trajectories, trust_scores

    def _get_end_effector_positions(self, trajectory):
        """Get end-effector positions for a given trajectory."""
        positions = []
        end_effector_link_index = self.sim.get_end_effector_link_index()
        for action in trajectory:
            self.sim.set_joint_positions(action)
            position = p.getLinkState(self.sim.robotId, end_effector_link_index)[0]
            positions.append(position)
        return positions

    def _get_end_effector_positions_from_qpos_sequence(self, qpos_sequence):
        """Get end-effector positions from a sequence of normalized qpos."""
        positions = []
        end_effector_link_index = self.sim.get_end_effector_link_index()
        for qpos_norm in qpos_sequence:
            qpos = self.unnormalize_qpos(qpos_norm)
            self.sim.set_joint_positions(qpos)
            position = p.getLinkState(self.sim.robotId, end_effector_link_index)[0]
            positions.append(position)
        return positions

    def _draw_trajectory(self, positions, color):
        """Draw the trajectory in the simulation."""
        for i in range(len(positions) - 1):
            self.draw_trajectory_segment(positions[i], positions[i + 1], color=color)

    def draw_trajectory_segment(self, start_pos, end_pos, color):
        """Draw a single segment of the trajectory in the simulation."""
        p.addUserDebugLine(
            start_pos,
            end_pos,
            lineColorRGB=color,
            lineWidth=3.0,
            lifeTime=1000,
        )

    def _draw_initial_position_marker(self, position):
        """Draw a red sphere at the initial position."""
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[1, 0, 0, 1],
        )
        body_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position,
        )
        return body_id

    def plot_trust_rewards(self, trust_rewards_list, save_path="trust_reward_plot.png"):
        """Plot and save trust rewards for multiple trajectories."""
        plt.figure(figsize=(10, 6))
        for idx, trust_rewards in enumerate(trust_rewards_list):
            plt.plot(
                trust_rewards, marker="o", linestyle="-", label=f"Trajectory {idx}"
            )
        plt.title("Trust Reward Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Trust Reward")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"Trust reward plot saved to {save_path}")
        plt.close()

    def visualize_policy_trajectories(
        self,
        policy_trajectories,
        images,
        actual_qpos_sequence=None,
        initial_qpos_norm=None,
        pause_time: float = 1 / 30.0,
    ):
        """Visualize multiple trajectories generated by the policy with the ability to replay."""
        logger.info(
            "Visualizing multiple policy trajectories with pre-drawn end-effector paths..."
        )

        exit_simulation = False
        marker_id = None

        while not exit_simulation:
            initial_position = policy_trajectories[0][0]
            self.sim.reset_to_position(initial_position)

            if initial_qpos_norm is not None:
                initial_qpos = self.unnormalize_qpos(initial_qpos_norm)
                self.sim.set_joint_positions(initial_qpos)
                end_effector_link_index = self.sim.get_end_effector_link_index()
                initial_ee_position = p.getLinkState(
                    self.sim.robotId, end_effector_link_index
                )[0]

                if marker_id is not None:
                    p.removeBody(marker_id)

                marker_id = self._draw_initial_position_marker(initial_ee_position)
                self.sim.reset_to_position(initial_position)

            for idx, trajectory in enumerate(policy_trajectories):
                predicted_positions = self._get_end_effector_positions(trajectory)
                color = [np.random.rand(), np.random.rand(), np.random.rand()]
                self._draw_trajectory(predicted_positions, color=color)

            if actual_qpos_sequence is not None:
                actual_positions = self._get_end_effector_positions_from_qpos_sequence(
                    actual_qpos_sequence
                )
                self._draw_trajectory(actual_positions, color=[0, 0, 1])

            frame_duration = pause_time

    def visualize_policy_trajectories(
        self,
        policy_trajectories,
        images,
        actual_qpos_sequence=None,
        initial_qpos_norm=None,
        pause_time: float = 1 / 30.0,
    ):
        """Visualize multiple trajectories generated by the policy with the ability to replay."""
        logger.info(
            "Visualizing multiple policy trajectories with pre-drawn end-effector paths..."
        )

        exit_simulation = False
        marker_id = None

        while not exit_simulation:
            initial_position = policy_trajectories[0][0]
            self.sim.reset_to_position(initial_position)

            if initial_qpos_norm is not None:
                initial_qpos = self.unnormalize_qpos(initial_qpos_norm)
                self.sim.set_joint_positions(initial_qpos)
                end_effector_link_index = self.sim.get_end_effector_link_index()
                initial_ee_position = p.getLinkState(
                    self.sim.robotId, end_effector_link_index
                )[0]

                if marker_id is not None:
                    p.removeBody(marker_id)

                marker_id = self._draw_initial_position_marker(initial_ee_position)
                self.sim.reset_to_position(initial_position)

            for idx, trajectory in enumerate(policy_trajectories):
                predicted_positions = self._get_end_effector_positions(trajectory)
                color = [np.random.rand(), np.random.rand(), np.random.rand()]
                self._draw_trajectory(predicted_positions, color=color)

            if actual_qpos_sequence is not None:
                actual_positions = self._get_end_effector_positions_from_qpos_sequence(
                    actual_qpos_sequence
                )
                self._draw_trajectory(actual_positions, color=[0, 0, 1])

            frame_duration = pause_time
            next_frame_time = time.time() + frame_duration

            first_trajectory = policy_trajectories[0]
            for step, action in enumerate(
                tqdm(first_trajectory, desc="Playing first policy trajectory")
            ):
                self.sim.apply_action(action, step_time=pause_time)

                if step < len(images):
                    image = images[step]
                    self.display_image(image)

                sleep_time = next_frame_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                next_frame_time += frame_duration

            logger.info("Total playback completed.")

            logger.info("Press 'r' to replay, 'q' to exit...")
            while True:
                keys = p.getKeyboardEvents()
                if ord("r") in keys and keys[ord("r")] & p.KEY_WAS_TRIGGERED:
                    logger.info("Replaying trajectory...")
                    break
                if ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED:
                    exit_simulation = True
                    break
                time.sleep(0.1)

        if marker_id is not None:
            p.removeBody(marker_id)

        logger.info("Exiting visualization...")

    def run_visualization(
        self,
        episode_path: str,
        mode: VisualizationMode = VisualizationMode.RECORDED,
        timestamp: int = None,
        save_path: str = None,
    ):
        """
        Run visualization for a specified episode and optionally save trajectories.

        Args:
            episode_path (str): Path to the episode file
            mode (VisualizationMode): Visualization mode
            timestamp (int, optional): The timestamp to start visualization from
            save_path (str, optional): Path to save the trajectories NPZ file
        """
        logger.info("Starting visualization...")

        # Load the episode data
        qpos_sequence, actions, trust_sequence, images = self.load_episode(episode_path)

        if save_path is None:
            # Generate default save path based on episode path and timestamp
            episode_name = os.path.basename(os.path.normpath(episode_path))
            timestamp_str = f"_t{timestamp}" if timestamp is not None else ""
            save_path = os.path.join(
                os.path.dirname(episode_path),
                f"{episode_name}_trajectories{timestamp_str}.npz",
            )

        if mode == VisualizationMode.POLICY_FROM_TIMESTAMP:
            if timestamp is None:
                raise ValueError(
                    "Timestamp must be provided for POLICY_FROM_TIMESTAMP mode."
                )

            logger.info("=== Visualizing Policy Trajectories from Timestamp ===")
            initial_qpos_norm = qpos_sequence[timestamp]
            image = images[timestamp]

            actual_trust_sequence = trust_sequence[timestamp : timestamp + 4]
            actual_trust_sequence = (
                torch.tensor(actual_trust_sequence).float().to(self.device).unsqueeze(0)
            )

            # Generate policy trajectories from current state
            policy_trajectories, trust_scores = self.generate_policy_trajectories(
                initial_qpos_norm, image, actual_trust_sequence
            )

            # Get actual positions for comparison
            traj_length = len(policy_trajectories[0])
            actual_qpos_sequence = qpos_sequence[timestamp : timestamp + traj_length]

            # Save trajectories before visualization
            self.save_trajectories_to_npz(
                policy_trajectories,
                actual_qpos_sequence=actual_qpos_sequence,
                save_path=save_path,
            )

            # Continue with visualization
            self.visualize_policy_trajectories(
                policy_trajectories,
                images[timestamp : timestamp + traj_length],
                actual_qpos_sequence=actual_qpos_sequence,
                initial_qpos_norm=initial_qpos_norm,
            )
            self.plot_trust_rewards(trust_scores)

        # Handle other visualization modes here if needed
        elif mode == VisualizationMode.RECORDED:
            logger.info("=== Visualizing Recorded Trajectory ===")
            self.visualize_recorded_trajectory(qpos_sequence, images)

        else:
            logger.error(f"Unsupported visualization mode: {mode}")

        logger.info("Visualization completed. Press Ctrl+C to exit...")
        try:
            while True:
                p.stepSimulation()
                time.sleep(1 / 240.0)
        except KeyboardInterrupt:
            p.disconnect()
            cv2.destroyAllWindows()


# Base checkpoint directory
CHECKPOINT_DIR = "<path_to_checkpoint_dir>"

# Derived paths
CONFIG_PATH = os.path.join(CHECKPOINT_DIR, "config")
STATS_PATH = os.path.join(CHECKPOINT_DIR, "stats.yaml")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "<checkpoint_name>.ckpt")
URDF_PATH = "robot_description/urdf/amiga/armmp.urdf"


@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.train.seed)
    # override mask_p in config
    cfg.policy.mask_p = 0.0

    # Paths
    urdf_path = URDF_PATH
    stats_path = STATS_PATH
    checkpoint_path = CHECKPOINT_PATH
    data_dir = cfg.task.dataset_dir

    # Specify the episode number and timestamp
    episode_number = 0
    timestamp = 0

    # Create save path for trajectories
    save_dir = os.path.join("trajectory_analysis")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        f"episode_{episode_number}_t{timestamp}_trajectories.npz",
    )

    # Update to use episode directory
    episode_dir = os.path.join(data_dir, f"episode_{episode_number}")

    if not os.path.exists(episode_dir):
        logger.error(f"Episode directory not found: {episode_dir}")
        return

    # Initialize visualizer
    visualizer = DatasetPolicyVisualizer(urdf_path, stats_path, checkpoint_path, cfg)
    visualizer.cameras = cfg.policy.camera_names

    # Run visualization with save path
    visualizer.run_visualization(
        episode_dir,
        mode=VisualizationMode.RECORDED,  # or VisualizationMode.POLICY_FROM_TIMESTAMP
        timestamp=timestamp,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
