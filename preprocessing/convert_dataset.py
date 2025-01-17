import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import yaml
import glob
import pickle
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from multiprocessing import Pool, cpu_count

from preprocessing.utils import (
    plot_in_grid,
    resize,
    shifted_center_crop,
    aligned_crop,
    create_video_ffmpeg,
)


def normalize_sequence_data(sequence_data, aggregated_mean, aggregated_std):
    """Normalize sequence data using precomputed statistics."""

    normalized_data = {}
    for key, value in sequence_data.items():
        if "rgb" not in key and "depth" not in key:
            if "trust" in key:
                normalized_data[key] = value / 1023.0
            else:
                mean = aggregated_mean.get(
                    f"observations/{key}" if key != "actions" else "action"
                )
                std = aggregated_std.get(
                    f"observations/{key}" if key != "actions" else "action"
                )
                if mean is not None and std is not None:
                    normalized_data[key] = (value - mean) / std
                else:
                    normalized_data[key] = value
        else:
            normalized_data[key] = value
    return normalized_data


def aggregate_stats(episode_dirs, trim_start=0, trim_end=0):
    """Aggregate statistics for normalization."""

    aggregated_stats = {}
    for episode_dir in tqdm(episode_dirs, desc="Aggregating stats"):
        frame_dirs = natsorted(
            [d for d in glob.glob(os.path.join(episode_dir, "*")) if os.path.isdir(d)]
        )

        if trim_end > 0:
            frame_dirs = frame_dirs[trim_start:-trim_end]
        else:
            frame_dirs = frame_dirs[trim_start:]

        previous_joint_positions = None
        previous_ee_pos_quat = None
        for frame_dir in frame_dirs:
            meta_path = os.path.join(frame_dir, "meta.pkl")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, "rb") as f:
                meta_data = pickle.load(f)
            current_joint_positions = meta_data.get("joint_positions")
            current_ee_pos_quat = meta_data.get("ee_pos_quat")
            data_dict = {}

            # Calculate diffs
            if (
                previous_joint_positions is not None
                and current_joint_positions is not None
            ):
                joint_diff = np.subtract(
                    current_joint_positions, previous_joint_positions
                )
                data_dict["observations/joint_position_diff"] = joint_diff
            if previous_ee_pos_quat is not None and current_ee_pos_quat is not None:
                ee_diff = np.subtract(current_ee_pos_quat, previous_ee_pos_quat)
                data_dict["observations/ee_pos_quat_diff"] = ee_diff
            previous_joint_positions = current_joint_positions
            previous_ee_pos_quat = current_ee_pos_quat

            for key in [
                "action",
                "joint_positions",
                "joint_velocities",
                "ee_pos_quat",
                "ee_vel",
                "ee_force",
                "gripper_position",
                "trust_level",
            ]:
                value = meta_data.get(key)
                if value is not None:
                    if key == "action":
                        data_dict["action"] = value
                    else:
                        data_dict[f"observations/{key}"] = value

            for key, value in data_dict.items():
                if "rgb" in key or "depth" in key:
                    continue  # Skip image data
                if key not in aggregated_stats:
                    aggregated_stats[key] = {
                        "count": 0,
                        "mean": np.zeros_like(value, dtype=np.float64),
                        "M2": np.zeros_like(value, dtype=np.float64),
                        "max": np.full_like(value, -np.inf, dtype=np.float64),
                        "min": np.full_like(value, np.inf, dtype=np.float64),
                    }
                stats = aggregated_stats[key]
                stats["count"] += 1
                delta = value - stats["mean"]
                stats["mean"] += delta / stats["count"]
                delta2 = value - stats["mean"]
                stats["M2"] += delta * delta2
                stats["max"] = np.maximum(stats["max"], value)
                stats["min"] = np.minimum(stats["min"], value)

    aggregated_max = {}
    aggregated_min = {}
    aggregated_mean = {}
    aggregated_std = {}
    for key, stats in aggregated_stats.items():
        count = stats["count"]
        mean = stats["mean"]
        variance = stats["M2"] / count if count > 1 else np.zeros_like(mean)
        std = np.sqrt(variance)
        std = np.clip(std, 1e-2, np.inf)  # Avoid division by zero
        aggregated_max[key] = stats["max"]
        aggregated_min[key] = stats["min"]
        aggregated_mean[key] = mean
        aggregated_std[key] = std
    return aggregated_max, aggregated_min, aggregated_mean, aggregated_std


def process_episode(args):
    """Process an episode and save sequence data and images."""

    (
        idx,
        episode_dir,
        image_size,
        output_dir,
        stride,
        save_depth,
        visualize,
        aggregated_mean,
        aggregated_std,
        trim_start,
        trim_end,
    ) = args
    output_episode_dir = os.path.join(output_dir, f"episode_{idx}")
    frames_dir = os.path.join(output_episode_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    frame_dirs = natsorted(
        [d for d in glob.glob(os.path.join(episode_dir, "*")) if os.path.isdir(d)]
    )

    episode_metadata = {
        "source": {
            "raw_episode_dir": os.path.basename(episode_dir),
            "full_path": episode_dir,
            "num_original_frames": len(frame_dirs),
            "num_processed_frames": len(
                frame_dirs[trim_start:-trim_end]
                if trim_end > 0
                else frame_dirs[trim_start:]
            ),
        }
    }

    # Save metadata
    metadata_path = os.path.join(output_episode_dir, "metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.dump(episode_metadata, f, default_flow_style=False)

    if trim_end > 0:
        frame_dirs = frame_dirs[trim_start:-trim_end]
    else:
        frame_dirs = frame_dirs[trim_start:]

    sequence_data = {
        "actions": [],
        "joint_positions": [],
        "joint_velocities": [],
        "ee_pos_quats": [],
        "ee_vels": [],
        "ee_forces": [],
        "gripper_positions": [],
        "trust_levels": [],
    }

    # For each frame
    for i, frame_dir in enumerate(frame_dirs):
        meta_path = os.path.join(frame_dir, "meta.pkl")
        if not os.path.exists(meta_path):
            continue

        with open(meta_path, "rb") as f:
            meta_data = pickle.load(f)

        # Collect sequence data
        sequence_data["actions"].append(meta_data.get("action"))
        sequence_data["joint_positions"].append(meta_data.get("joint_positions"))
        sequence_data["joint_velocities"].append(meta_data.get("joint_velocities"))
        sequence_data["ee_pos_quats"].append(meta_data.get("ee_pos_quat"))
        sequence_data["ee_vels"].append(meta_data.get("ee_vel"))
        sequence_data["ee_forces"].append(meta_data.get("ee_force"))
        sequence_data["gripper_positions"].append(meta_data.get("gripper_position"))
        sequence_data["trust_levels"].append(meta_data.get("trust_level"))

        # Save images at fixed stride intervals
        if i % stride == 0:
            frame_idx_str = f"{i:06d}"
            frame_output_dir = os.path.join(frames_dir, frame_idx_str)
            os.makedirs(frame_output_dir, exist_ok=True)

            # Process images
            for camera in ["wrist", "base"]:
                rgb_filename = f"{camera}_rgb.jpg"
                depth_filename = f"{camera}_depth.png"
                rgb_path = os.path.join(frame_dir, rgb_filename)
                depth_path = os.path.join(frame_dir, depth_filename)

                if os.path.exists(rgb_path):
                    rgb_image = cv2.imread(rgb_path)
                    if rgb_image is None:
                        continue
                    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                    rgb_image = rgb_image.transpose(2, 0, 1)  # HWC to CHW

                    depth_image = None
                    if os.path.exists(depth_path):
                        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                        if depth_image is not None:
                            depth_image = depth_image[
                                np.newaxis, :, :
                            ]  # Add channel dimension

                    # Process images based on camera - for wrist camera, apply a shifted center crop
                    if camera == "wrist":
                        rgb_image, depth_image = resize(
                            *shifted_center_crop(
                                rgb_image, depth_image, shift_x=100, shift_y=0
                            ),
                            size=image_size,
                        )
                    else:
                        rgb_image, depth_image = resize(
                            *aligned_crop(rgb_image, depth_image, alignment="center"),
                            size=image_size,
                        )

                    rgb_image = rgb_image.transpose(1, 2, 0)  # CHW to HWC
                    rgb_save_path = os.path.join(frame_output_dir, f"{camera}_rgb.jpg")
                    cv2.imwrite(
                        rgb_save_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                    )

                    if save_depth and depth_image is not None:
                        depth_image = depth_image[0]  # Remove channel dimension
                        depth_save_path = os.path.join(
                            frame_output_dir, f"{camera}_depth.png"
                        )
                        cv2.imwrite(depth_save_path, depth_image)

    for key in sequence_data:
        sequence_data[key] = np.array(sequence_data[key])

    sequence_data = normalize_sequence_data(
        sequence_data, aggregated_mean, aggregated_std
    )

    sequence_save_path = os.path.join(output_episode_dir, "sequence.npz")
    np.savez_compressed(sequence_save_path, **sequence_data)

    if visualize:
        vis_dir = os.path.join(output_dir, "vis")
        rgb_output_dir = os.path.join(vis_dir, "rgb")
        depth_output_dir = os.path.join(vis_dir, "depth")
        action_output_dir = os.path.join(vis_dir, "actions")
        state_output_dir = os.path.join(vis_dir, "states")

        os.makedirs(rgb_output_dir, exist_ok=True)
        os.makedirs(depth_output_dir, exist_ok=True)
        os.makedirs(action_output_dir, exist_ok=True)
        os.makedirs(state_output_dir, exist_ok=True)

        for camera in ["wrist", "base"]:
            rgb_frames = []
            depth_frames = []

            frame_sample = cv2.imread(os.path.join(frame_dirs[0], f"{camera}_rgb.jpg"))
            if frame_sample is not None:
                rgb_sample = cv2.cvtColor(frame_sample, cv2.COLOR_BGR2RGB)
                rgb_sample = rgb_sample.transpose(2, 0, 1)  # HWC to CHW

                depth_sample_path = os.path.join(frame_dirs[0], f"{camera}_depth.png")
                depth_sample = None
                if os.path.exists(depth_sample_path):
                    depth_sample = cv2.imread(depth_sample_path, cv2.IMREAD_UNCHANGED)
                    if depth_sample is not None:
                        depth_sample = depth_sample[np.newaxis, :, :]

                if camera == "wrist":
                    rgb_sample, depth_sample = resize(
                        *shifted_center_crop(
                            rgb_sample, depth_sample, shift_x=100, shift_y=0
                        ),
                        size=376,
                    )
                else:
                    rgb_sample, depth_sample = resize(
                        *aligned_crop(rgb_sample, depth_sample, alignment="center"),
                        size=376,
                    )

                batch_size = 100
                for i in range(0, len(frame_dirs), batch_size):
                    batch_dirs = frame_dirs[i : i + batch_size]

                    for frame_dir in batch_dirs:
                        rgb_path = os.path.join(frame_dir, f"{camera}_rgb.jpg")
                        depth_path = os.path.join(frame_dir, f"{camera}_depth.png")

                        if os.path.exists(rgb_path):
                            rgb_img = cv2.imread(rgb_path)
                            if rgb_img is not None:
                                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                                rgb_img = rgb_img.transpose(2, 0, 1)  # HWC to CHW

                                depth_img = None
                                if os.path.exists(depth_path):
                                    depth_img = cv2.imread(
                                        depth_path, cv2.IMREAD_UNCHANGED
                                    )
                                    if depth_img is not None:
                                        depth_img = depth_img[np.newaxis, :, :]

                                if camera == "wrist":
                                    rgb_img, depth_img = resize(
                                        *shifted_center_crop(
                                            rgb_img, depth_img, shift_x=100, shift_y=0
                                        ),
                                        size=376,
                                    )
                                else:
                                    rgb_img, depth_img = resize(
                                        *aligned_crop(
                                            rgb_img, depth_img, alignment="center"
                                        ),
                                        size=376,
                                    )

                                rgb_img = rgb_img.transpose(1, 2, 0)
                                rgb_frames.append(
                                    cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                                )

                        if os.path.exists(depth_path) and depth_img is not None:
                            depth_img = depth_img[0]  # Remove channel dimension
                            depth_img = cv2.normalize(
                                depth_img, None, 0, 255, cv2.NORM_MINMAX
                            )
                            depth_img = cv2.cvtColor(
                                depth_img.astype(np.uint8), cv2.COLOR_GRAY2BGR
                            )
                            depth_frames.append(depth_img)

                if rgb_frames:
                    rgb_output_path = os.path.join(
                        rgb_output_dir, f"episode_{idx}_{camera}_rgb.mp4"
                    )
                    create_video_ffmpeg(rgb_frames, rgb_output_path, crf=28)

                if depth_frames:
                    depth_output_path = os.path.join(
                        depth_output_dir, f"episode_{idx}_{camera}_depth.mp4"
                    )
                    create_video_ffmpeg(depth_frames, depth_output_path, crf=28)

        if sequence_data["actions"] is not None and len(sequence_data["actions"]) > 0:
            action_plot_path = os.path.join(
                action_output_dir, f"episode_{idx}_actions.png"
            )
            plot_in_grid([sequence_data["actions"]], action_plot_path)

        for key in sequence_data:
            if (
                key not in ["actions"]
                and sequence_data[key] is not None
                and len(sequence_data[key]) > 0
            ):
                state_plot_path = os.path.join(
                    state_output_dir, f"episode_{idx}_{key}.png"
                )
                plot_in_grid([sequence_data[key]], state_plot_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw trajectory data to a normalized format."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to the source dataset directory.",
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="Stride interval for saving images"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=cpu_count() - 1,
        help="Number of multiprocessing workers",
    )
    parser.add_argument(
        "--save_depth", action="store_true", help="Whether to save depth images"
    )
    parser.add_argument(
        "--image_size", type=int, default=224, help="Size of the output images"
    )
    parser.add_argument(
        "--visualize",
        type=bool,
        default=True,
        help="Enable visualization support (True/False)",
    )
    parser.add_argument(
        "--trim_start",
        type=int,
        default=5,
        help="Number of frames to trim from start of sequence",
    )
    parser.add_argument(
        "--trim_end",
        type=int,
        default=5,
        help="Number of frames to trim from end of sequence",
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    stride = args.stride
    num_workers = args.num_workers
    save_depth = args.save_depth
    image_size = args.image_size
    visualize = args.visualize
    trim_start = args.trim_start
    trim_end = args.trim_end

    output_dir = os.path.join(
        os.path.dirname(source_dir),
        os.path.basename(source_dir) + f"_processed_{image_size}px_s{stride}",
    )
    if os.path.isdir(output_dir):
        user_input = input(
            f"Output directory {output_dir} already exists. Do you want to overwrite it? [y/N]: "
        )
        if user_input.lower() != "y":
            print("Aborting...")
            return

        print(f"Cleaning non-vis contents from {output_dir}")
        vis_dir = os.path.join(output_dir, "vis")
        if os.path.exists(vis_dir):
            temp_vis_dir = vis_dir + "_temp"
            shutil.move(vis_dir, temp_vis_dir)
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        if "temp_vis_dir" in locals():
            shutil.move(temp_vis_dir, vis_dir)
    else:
        os.makedirs(output_dir)

    episode_dirs = natsorted(
        [d for d in glob.glob(os.path.join(source_dir, "*")) if os.path.isdir(d)]
    )

    print("Computing normalization statistics...")
    aggregated_max, aggregated_min, aggregated_mean, aggregated_std = aggregate_stats(
        episode_dirs, trim_start=trim_start, trim_end=trim_end
    )

    stats_dict = {
        "max": {k: v.tolist() for k, v in aggregated_max.items()},
        "min": {k: v.tolist() for k, v in aggregated_min.items()},
        "mean": {k: v.tolist() for k, v in aggregated_mean.items()},
        "std": {k: v.tolist() for k, v in aggregated_std.items()},
    }

    stats_path = os.path.join(output_dir, "stats.yaml")
    with open(stats_path, "w") as f:
        yaml.dump(stats_dict, f, default_flow_style=False)

    dataset_info = {
        "num_episodes": len(episode_dirs),
        "stride": stride,
        "save_depth": save_depth,
        "image_size": image_size,
        "trim_start": trim_start,
        "trim_end": trim_end,
    }
    dataset_info_path = os.path.join(output_dir, "dataset_info.yaml")
    with open(dataset_info_path, "w") as f:
        yaml.dump(dataset_info, f, default_flow_style=False)

    print("Processing episodes...")
    pool = Pool(num_workers)
    args_list = [
        (
            idx,
            episode_dir,
            image_size,
            output_dir,
            stride,
            save_depth,
            visualize,
            aggregated_mean,
            aggregated_std,
            trim_start,
            trim_end,
        )
        for idx, episode_dir in enumerate(episode_dirs)
    ]
    list(tqdm(pool.imap_unordered(process_episode, args_list), total=len(episode_dirs)))
    pool.close()
    pool.join()
    print("Processing complete.")

    if visualize:
        # Create video grids for each camera and modality
        print("Creating video grids...")
        vis_dir = os.path.join(output_dir, "vis")
        grid_dir = os.path.join(vis_dir, "grids")
        os.makedirs(grid_dir, exist_ok=True)

        from video_grid import VideoGridCreator

        for camera in ["wrist", "base"]:
            for modality in ["rgb", "depth"]:
                print(
                    f"Creating video grid for {camera} camera, {modality} modality..."
                )
                creator = VideoGridCreator()
                try:
                    creator.create_video_grid(
                        folder_path=vis_dir,
                        output_filename=os.path.join(
                            grid_dir, f"grid_{camera}_{modality}.mp4"
                        ),
                        camera=camera,
                        modality=modality,
                        video_dimensions=(224, 224),
                        grid_size=(7, 6),
                        overwrite=True,
                    )
                except Exception as e:
                    print(
                        f"Error creating video grid for {camera} {modality}: {str(e)}"
                    )

        print("Video grid creation complete.")


if __name__ == "__main__":
    main()
