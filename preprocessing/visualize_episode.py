import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import cv2
from collections import defaultdict
import argparse


def load_episode(episode_dir):
    """Load episode data from NPZ file."""
    sequence_path = os.path.join(episode_dir, "sequence.npz")
    with np.load(sequence_path) as data:
        return {key: data[key] for key in data.files}


def preload_video(video_path, max_frames=None):
    """Preload all frames from a video."""
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    for _ in range(total_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            break

    cap.release()
    return frames


def visualize_episode(episode_dir, vis_dir):
    """Visualize episode data with synchronized videos."""

    # Load sequence data
    episode_data = load_episode(episode_dir)
    episode_idx = os.path.basename(episode_dir).split("_")[1]

    # Preload all video frames
    print("Preloading video frames...")
    video_frames = {}
    for camera in ["wrist", "base"]:
        rgb_path = os.path.join(
            vis_dir, "rgb", f"episode_{episode_idx}_{camera}_rgb.mp4"
        )
        depth_path = os.path.join(
            vis_dir, "depth", f"episode_{episode_idx}_{camera}_depth.mp4"
        )
        if os.path.exists(rgb_path) and os.path.exists(depth_path):
            video_frames[f"{camera}_rgb"] = preload_video(
                rgb_path, len(episode_data["actions"])
            )
            video_frames[f"{camera}_depth"] = preload_video(
                depth_path, len(episode_data["actions"])
            )
    print("Video frames loaded!")

    # List of datasets to plot
    dataset_list = [
        ("actions", "Robot Actions"),
        ("joint_positions", "Joint Positions"),
        ("trust_levels", "Trust Level"),
    ]

    n_images = len(video_frames)
    n_datasets = len(dataset_list)

    # Create main figure and GridSpec
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 1, height_ratios=[1, 3], figure=fig)

    # Sub-GridSpec for images
    gs_images = GridSpecFromSubplotSpec(1, n_images, subplot_spec=gs[0], wspace=0.05)

    # Axes for images
    image_axes = []
    image_plots = []
    for i in range(n_images):
        ax = fig.add_subplot(gs_images[0, i])
        ax.axis("off")
        image_axes.append(ax)

    # Sub-GridSpec for plots
    gs_plots = GridSpecFromSubplotSpec(n_datasets, 1, subplot_spec=gs[1], hspace=0.5)

    vline_list = []
    plot_lines = defaultdict(list)

    # Plot each dataset
    for idx, (dataset_name, dataset_title) in enumerate(dataset_list):
        ax = fig.add_subplot(gs_plots[idx, 0])
        data = episode_data[dataset_name]

        if data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1):
            (line,) = ax.plot(np.arange(len(data)), data.squeeze(), label=dataset_title)
            plot_lines[dataset_name].append(line)
        else:
            for i in range(data.shape[1]):
                (line,) = ax.plot(
                    np.arange(len(data)), data[:, i], label=f"{dataset_title} [{i}]"
                )
                plot_lines[dataset_name].append(line)

        ax.legend(loc="upper right", fontsize=8, ncol=3)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Value")
        ax.set_title(dataset_title, fontsize=12)
        ax.set_xlim(0, len(data) - 1)
        ax.set_ylim(np.min(data), np.max(data))

        vline = ax.axvline(x=0, color="gray", linestyle="--")
        vline_list.append(vline)

    # Initial display of the first frame
    for i, (key, frames) in enumerate(video_frames.items()):
        if frames:
            img = image_axes[i].imshow(frames[0])
            image_axes[i].set_title(key.replace("_", " ").title(), fontsize=12)
            image_plots.append(img)

    def update(val):
        step = int(slider.val)

        # Update images using pre-loaded frames
        for i, (key, frames) in enumerate(video_frames.items()):
            if frames and step < len(frames):
                image_plots[i].set_array(frames[step])

        # Update vertical lines in plots
        for vline in vline_list:
            vline.set_xdata([step, step])

    # Slider for frame navigation
    slider_ax = fig.add_axes([0.2, 0.01, 0.6, 0.02])
    slider = Slider(
        slider_ax,
        "Frame",
        0,
        len(episode_data["actions"]) - 1,
        valinit=0,
        valstep=1,
        valfmt="%d",
    )
    slider.on_changed(update)

    # Play/Pause button
    play_ax = fig.add_axes([0.85, 0.01, 0.1, 0.04])
    play_button = Button(play_ax, "Play")

    # Use a more efficient timer for real-time playback
    timer = fig.canvas.new_timer(interval=33.33)  # Exactly 30 fps

    def advance_frame():
        current_frame = int(slider.val)
        max_frame = int(slider.valmax)
        if current_frame < max_frame:
            slider.set_val(current_frame + 1)
        else:
            slider.set_val(0)
            timer.stop()
            play_button.label.set_text("Play")

    timer.add_callback(advance_frame)

    is_playing = False

    def on_play_pause(event):
        nonlocal is_playing
        if is_playing:
            is_playing = False
            play_button.label.set_text("Play")
            timer.stop()
        else:
            is_playing = True
            play_button.label.set_text("Pause")
            timer.start()

    play_button.on_clicked(on_play_pause)

    # More efficient layout
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])

    # Enable blitting for more efficient animation
    fig.canvas.draw()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize robot episode data with synchronized videos."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing the dataset",
    )
    parser.add_argument(
        "--episode", type=int, required=True, help="Episode number to visualize"
    )
    args = parser.parse_args()

    episode_dir = os.path.join(args.base_dir, f"episode_{args.episode}")
    vis_dir = os.path.join(args.base_dir, "vis")

    if not os.path.exists(episode_dir):
        raise ValueError(f"Episode directory does not exist: {episode_dir}")
    if not os.path.exists(vis_dir):
        raise ValueError(f"Visualization directory does not exist: {vis_dir}")

    visualize_episode(episode_dir, vis_dir)


if __name__ == "__main__":
    main()
