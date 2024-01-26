import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from vslam.utils import load_poses
from tqdm import tqdm
from vslam.definitions import *

def plot_trajectories_animation(real_trajectory_path, trajectory_files, labels, sequence_number, output_video, fps=30):
    """
    Create a video animation showing the progression of multiple trajectories against the real trajectory.

    :param real_trajectory_path: Path to the real trajectory file.
    :param trajectory_files: List of file paths to the trajectories to plot.
    :param labels: List of labels for each trajectory.
    :param sequence_number: Data track identifier for the real trajectory.
    :param output_video: Path to save the output video.
    :param fps: Frames per second for the output video.
    """
    # Load real trajectory
    real_trajectory = load_poses(real_trajectory_path)
    kitti_positions = np.array([pose[:3, 3] for pose in real_trajectory])

    # Load other trajectories
    trajectories = [np.array([pose[:3, 3] for pose in np.load(file)]) for file in trajectory_files]

    # Determine the maximum length of the trajectories
    max_length = max(len(kitti_positions), max(len(traj) for traj in trajectories))

    # Setup the plot
    dpi = 300  # Adjust as needed for the desired physical size of the figure
    fig, ax = plt.subplots(figsize=(2560/dpi, 1449/dpi), dpi=dpi)
    ax.set_title('Trajectory Comparison')
    ax.grid(True)
    ax.set_xlabel('X position')
    ax.set_ylabel('Z position')

    # Create and update the progress bar
    pbar = tqdm(total=max_length, desc="Rendering Animation")

    # Plotting function for each frame
    def update(num, kitti_positions, trajectories, labels):
        ax.clear()
        ax.plot(kitti_positions[:num, 0], kitti_positions[:num, 2], label=f'KITTI seq{sequence_number} Ground Truth')
        for traj, label in zip(trajectories, labels):
            ax.plot(traj[:num, 0], traj[:num, 2], label=label, linestyle='--')
        ax.legend()
        ax.grid(True)
        pbar.update(1)

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=max_length, fargs=(kitti_positions, trajectories, labels), interval=1000/fps)

    # Adjust the writer settings for 4K output
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(output_video, writer=writer, dpi=dpi)

    # Close the progress bar
    pbar.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a video animation of multiple trajectories against a real trajectory.")
    parser.add_argument("--sequence_number", help="Data track identifier for the real trajectory.")
    parser.add_argument("--trajectory_files", nargs='+', help="File paths to the trajectories to plot.", required=True)
    parser.add_argument("--labels", nargs='+', help="Labels for each trajectory.", required=True)
    parser.add_argument("--output_video", help="Path to save the output video.", required=True)
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video.")

    args = parser.parse_args()

    real_trajectory_path = f'{DATASET_PATH}/poses/{args.sequence_number}.txt'
    plot_trajectories_animation(real_trajectory_path, args.trajectory_files, args.labels, args.sequence_number, args.output_video, args.fps)
