import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tempfile
import shutil
import pickle
import numpy as np

from preprocessing.convert_dataset import aggregate_stats


def test_aggregate_stats():
    # Create a temporary directory to simulate the dataset
    temp_dir = tempfile.mkdtemp()

    try:
        # Create episode directories
        episode_dirs = []
        for ep_idx in range(2):  # Two episodes
            episode_dir = os.path.join(temp_dir, f"episode_{ep_idx}")
            os.makedirs(episode_dir)
            episode_dirs.append(episode_dir)

            # Create frame directories with meta.pkl files
            for frame_idx in range(3):  # Three frames per episode
                frame_dir = os.path.join(episode_dir, f"frame_{frame_idx}")
                os.makedirs(frame_dir)
                meta_path = os.path.join(frame_dir, "meta.pkl")

                # Create mock joint_positions data
                joint_positions = np.array(
                    [
                        frame_idx,
                        frame_idx + 1,
                        frame_idx + 2,
                        frame_idx + 3,
                        frame_idx + 4,
                        frame_idx + 5,
                        frame_idx + 6,
                        frame_idx + 7,
                    ],
                    dtype=np.float32,
                )
                meta_data = {
                    "joint_positions": joint_positions,
                    "ee_pos_quat": None,  # Simplify for this test
                    # Add other necessary keys if needed
                }
                with open(meta_path, "wb") as f:
                    pickle.dump(meta_data, f)

        # Run aggregate_stats on the episode_dirs
        aggregated_max, aggregated_min, aggregated_mean, aggregated_std = (
            aggregate_stats(episode_dirs)
        )

        # Collect all joint_positions to compute expected values
        all_joint_positions = []
        for ep_idx in range(2):
            for frame_idx in range(3):
                joint_positions = np.array(
                    [
                        frame_idx,
                        frame_idx + 1,
                        frame_idx + 2,
                        frame_idx + 3,
                        frame_idx + 4,
                        frame_idx + 5,
                        frame_idx + 6,
                        frame_idx + 7,
                    ],
                    dtype=np.float32,
                )
                all_joint_positions.append(joint_positions)
        all_joint_positions = np.array(all_joint_positions)

        # Calculate expected mean and standard deviation
        expected_mean = np.mean(all_joint_positions, axis=0)
        expected_std = np.std(
            all_joint_positions, axis=0, ddof=0
        )  # ddof=0 for population std

        # Compare with aggregated_mean and aggregated_std from your function
        key = "observations/joint_positions"
        calculated_mean = aggregated_mean[key]
        calculated_std = aggregated_std[key]

        # Assertions to check if the values match
        assert np.allclose(
            calculated_mean, expected_mean
        ), f"Mean mismatch: {calculated_mean} vs {expected_mean}"
        assert np.allclose(
            calculated_std, expected_std
        ), f"Std dev mismatch: {calculated_std} vs {expected_std}"

        print(
            "Test passed: aggregate_stats computes mean and std dev correctly for joint_positions"
        )

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)


# Run the test
if __name__ == "__main__":
    test_aggregate_stats()
