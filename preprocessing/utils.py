import os
import cv2
import subprocess
import numpy as np
import matplotlib.pyplot as plt


def create_video_ffmpeg(frames, output_path, fps=30, crf=28):
    """
    Create video using FFmpeg with better compression, suppressing output.
    crf: Constant Rate Factor (0-51), lower means better quality, 23 is default
    """
    command = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-r",
        str(fps),
        "-i",
        "-",  # Input from pipe
        "-c:v",
        "libx264",
        "-preset",
        "slower",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]

    # Redirect both stdout and stderr to devnull to suppress output
    with open(os.devnull, "w") as devnull:
        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=devnull, stderr=devnull
        )

        for frame in frames:
            success, encoded_image = cv2.imencode(".png", frame)
            if success:
                pipe.stdin.write(encoded_image.tobytes())

        pipe.stdin.close()
        pipe.wait()


def plot_in_grid(vals: np.ndarray, save_path: str):
    """Plot the trajectories in a grid.

    Args:
        vals: List of arrays, each with shape T x N, where
            T is the number of timesteps,
            N is the dimensionality of the values.
        save_path: path to save the plot.
    """
    plt.clf()  # Clear current figure
    B = len(vals)
    N = vals[0].shape[-1]

    # Calculate the number of rows and columns needed
    cols = min(N, 4)  # Max 4 columns
    rows = (N + cols - 1) // cols  # Ceiling division to get number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)

    for b in range(B):
        curr = vals[b]
        T = curr.shape[0]
        for i in range(N):
            row, col = divmod(i, cols)
            ax = axes[row, col]
            ax.plot(np.arange(T), curr[:, i], alpha=0.5)
            ax.set_title(f"Dim {i}")

    # Remove any unused subplots
    for i in range(N, rows * cols):
        row, col = divmod(i, cols)
        if row < axes.shape[0] and col < axes.shape[1]:
            fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")
    del fig, axes

    # Only create 3D plot if N >= 3
    if N >= 3:
        plt.clf()
        fig = plt.figure(figsize=(20, 5))
        views = [(270, 0), (0, 0), (0, 90), (30, 45)]  # Different view angles

        for i, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(1, 4, i + 1, projection="3d")
            for b in range(B):
                curr = vals[b]
                ax.plot(
                    curr[:, 0], curr[:, 1], curr[:, 2], alpha=0.75, label="Trajectory"
                )
                ax.scatter(curr[0, 0], curr[0, 1], curr[0, 2], c="r", label="Start")
                ax.scatter(curr[-1, 0], curr[-1, 1], curr[-1, 2], c="g", label="End")

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev, azim)
            if i == 0:
                ax.legend()

        if "joint_position" in save_path or "ee_pos" in save_path:
            if "diff" not in save_path:
                plt.savefig(save_path[:-4] + "_3d.png")
        plt.close("all")
        plt.clf()
        del fig, ax


def shifted_center_crop(rgb_frame, depth_frame, shift_x=0, shift_y=0):
    """
    Crop frames to a square with an offset from center.

    Args:
        rgb_frame: RGB frame with shape (C, H, W)
        depth_frame: Depth frame with shape (1, H, W)
        shift_x: Number of pixels to shift horizontally (positive = right, negative = left)
        shift_y: Number of pixels to shift vertically (positive = down, negative = up)

    Returns:
        Tuple of cropped (rgb_frame, depth_frame)
    """
    H, W = rgb_frame.shape[-2:]
    sq_size = min(H, W)

    if H > W:  # Tall image
        # Calculate center start point
        start_y = (H - sq_size) // 2
        # Apply vertical shift while ensuring we stay within bounds
        start_y = max(0, min(H - sq_size, start_y + shift_y))
        rgb_frame = rgb_frame[..., start_y : start_y + sq_size, :]
        depth_frame = depth_frame[..., start_y : start_y + sq_size, :]

    elif W > H:  # Wide image
        # Calculate center start point
        start_x = (W - sq_size) // 2
        # Apply horizontal shift while ensuring we stay within bounds
        start_x = max(0, min(W - sq_size, start_x + shift_x))
        rgb_frame = rgb_frame[..., :, start_x : start_x + sq_size]
        depth_frame = depth_frame[..., :, start_x : start_x + sq_size]

    return rgb_frame, depth_frame


def aligned_crop(rgb_frame, depth_frame, alignment="center"):
    """
    Crop frames to a square with specified alignment.

    Args:
        rgb_frame: RGB frame with shape (C, H, W)
        depth_frame: Depth frame with shape (1, H, W)
        alignment: One of 'left', 'right', 'center' (default: 'center')

    Returns:
        Tuple of cropped (rgb_frame, depth_frame)
    """
    if alignment not in ["left", "right", "center"]:
        raise ValueError("alignment must be one of 'left', 'right', 'center'")

    H, W = rgb_frame.shape[-2:]
    sq_size = min(H, W)

    if H > W:  # Tall image
        if alignment == "center":
            start = (H - sq_size) // 2
        elif alignment == "left":
            start = 0
        else:  # right
            start = H - sq_size

        rgb_frame = rgb_frame[..., start : start + sq_size, :]
        depth_frame = depth_frame[..., start : start + sq_size, :]

    elif W > H:  # Wide image
        if alignment == "center":
            start = (W - sq_size) // 2
        elif alignment == "left":
            start = 0
        else:  # right
            start = W - sq_size

        rgb_frame = rgb_frame[..., :, start : start + sq_size]
        depth_frame = depth_frame[..., :, start : start + sq_size]

    return rgb_frame, depth_frame


def resize(rgb, depth, size=224):
    """Resize the RGB and depth frames to a square of the specified size.

    Args:
        rgb: RGB frame with shape (C, H, W)
        depth: Depth frame with shape (1, H, W)
        size: Size of the square image (default: 224)

    Returns:
        Tuple of resized (rgb_frame, depth_frame)
    """
    rgb = rgb.transpose([1, 2, 0])
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = rgb.transpose([2, 0, 1])

    depth = cv2.resize(depth[0], (size, size), interpolation=cv2.INTER_LINEAR)
    depth = depth.reshape([1, size, size])
    return rgb, depth
