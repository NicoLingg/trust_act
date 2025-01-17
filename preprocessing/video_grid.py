import os
import cv2
import glob
import random
import logging
import argparse
import subprocess
from typing import Tuple, List


class VideoGridCreator:
    """A class to create a grid of videos using ffmpeg."""

    def __init__(self):
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging for the application."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def _validate_inputs(
        self, folder_path: str, output_filename: str, grid_size: Tuple[int, int]
    ) -> None:
        """Validate inputs efficiently."""
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder path does not exist: {folder_path}")

        output_dir = os.path.dirname(os.path.abspath(output_filename))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not all(x > 0 for x in grid_size):
            raise ValueError("Grid dimensions must be positive integers")

        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError("ffmpeg is not installed or not accessible")

    def _get_video_files(
        self, folder_path: str, camera: str, modality: str
    ) -> List[str]:
        """Get video files efficiently using glob."""
        pattern = os.path.join(
            folder_path, modality, f"episode_*_{camera}_{modality}.mp4"
        )
        return sorted(glob.glob(pattern))

    def create_video_grid(
        self,
        folder_path: str,
        output_filename: str,
        camera: str,
        modality: str,
        video_dimensions: Tuple[int, int] = None,
        grid_size: Tuple[int, int] = (10, 6),
        overwrite: bool = False,
        random_seed: int = 41,
    ) -> None:
        """Create video grid with optimized processing."""
        self._validate_inputs(folder_path, output_filename, grid_size)

        if random_seed is not None:
            random.seed(random_seed)

        files = self._get_video_files(folder_path, camera, modality)
        if not files:
            raise ValueError(
                f"No matching video files found in {folder_path} for camera {camera} and modality {modality}"
            )

        # Get video dimensions once from the first video if not provided
        if video_dimensions is None:
            cap = cv2.VideoCapture(files[0])
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {files[0]}")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            video_width, video_height = width, height
            self.logger.info(f"Detected video dimensions: {video_width}x{video_height}")
        else:
            video_width, video_height = video_dimensions
            self.logger.info(
                f"Using manual video dimensions: {video_width}x{video_height}"
            )

        num_videos_per_row, max_rows = grid_size
        required_videos = num_videos_per_row * max_rows

        # Randomly select videos if we have more than needed
        if len(files) > required_videos:
            self.logger.info(
                f"Randomly selecting {required_videos} videos from {len(files)} available videos"
            )
            files = random.sample(files, required_videos)
            files.sort()

        total_videos = len(files)

        if total_videos < num_videos_per_row:
            raise ValueError(
                f"Not enough videos to fill one row (minimum {num_videos_per_row} required)"
            )

        # Generate filter complex string
        scale_filters = "".join(
            f"[{i}:v]scale={video_width}:{video_height}[v{i}];"
            for i in range(total_videos)
        )
        input_refs = "".join(f"[v{i}]" for i in range(total_videos))
        layout = "|".join(
            f"{i%num_videos_per_row*video_width}_{i//num_videos_per_row*video_height}"
            for i in range(total_videos)
        )
        filter_complex = f"{scale_filters}{input_refs}xstack=inputs={total_videos}:layout={layout}[out]"

        ffmpeg_cmd = [
            "ffmpeg",
            *(["-y"] if overwrite else []),
            *sum([["-i", file] for file in files[:total_videos]], []),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-s",
            f"{num_videos_per_row*video_width}x{(total_videos//num_videos_per_row)*video_height}",
            "-threads",
            "0",
            output_filename,
        ]

        self.logger.info("Starting ffmpeg process...")
        try:
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            _, stderr = process.communicate()

            if process.returncode != 0:
                raise subprocess.SubprocessError(
                    f"ffmpeg failed with return code {process.returncode}\nError: {stderr}"
                )

            self.logger.info("Successfully created video grid")

        except subprocess.SubprocessError as e:
            self.logger.error(f"Error during ffmpeg processing: {str(e)}")
            raise


def parse_args():
    """Parse command line arguments efficiently."""
    parser = argparse.ArgumentParser(
        description="Create a grid of videos from MP4 files in a folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-folder",
        "-i",
        required=True,
        help="Path to the folder containing MP4 files",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output.mp4",
        help="Output filename for the video grid",
    )
    parser.add_argument(
        "--camera",
        "-c",
        choices=["wrist", "base"],
        required=True,
        help="Camera view to use (wrist or base)",
    )
    parser.add_argument(
        "--modality",
        "-m",
        choices=["rgb", "depth"],
        required=True,
        help="Video modality to use (rgb or depth)",
    )
    parser.add_argument(
        "--grid-width",
        type=int,
        default=7,
        help="Number of videos per row in the grid",
    )
    parser.add_argument(
        "--grid-height",
        type=int,
        default=6,
        help="Maximum number of rows in the grid",
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=None,
        help="Width of each video in the grid (optional, will auto-detect if not specified)",
    )
    parser.add_argument(
        "--video-height",
        type=int,
        default=None,
        help="Height of each video in the grid (optional, will auto-detect if not specified)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite output file if it exists",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=41,
        help="Random seed for video selection",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        video_dimensions = None
        if args.video_width is not None and args.video_height is not None:
            video_dimensions = (args.video_width, args.video_height)

        creator = VideoGridCreator()
        creator.create_video_grid(
            folder_path=args.input_folder,
            output_filename=args.output,
            camera=args.camera,
            modality=args.modality,
            video_dimensions=video_dimensions,
            grid_size=(args.grid_width, args.grid_height),
            overwrite=args.force,
            random_seed=args.random_seed,
        )
    except Exception as e:
        logging.error(f"Failed to create video grid: {str(e)}")
        exit(1)
