import numpy as np
import matplotlib.pyplot as plt
import argparse
from vslam.utils import load_poses
from vslam.definitions import *

def plot_trajectories(real_trajectory_path, trajectory_files, labels, data_track):
    """
    Plot multiple trajectories against the real trajectory for comparison,
    along with X and Z velocities.

    :param real_trajectory_path: Path to the real trajectory file.
    :param trajectory_files: List of file paths to the trajectories to plot.
    :param labels: List of labels for each trajectory.
    :param data_track: Data track identifier for the real trajectory.
    """
    # Load real trajectory
    real_trajectory = load_poses(real_trajectory_path)
    kitti_positions = np.array([pose[:3, 3] for pose in real_trajectory])

    # Initialize figure and subplots
    fig = plt.figure(figsize=(15, 6))
    ax_trajectory = fig.add_subplot(1, 2, 1)
    ax_x_velocity = fig.add_subplot(2, 2, 2)
    ax_z_velocity = fig.add_subplot(2, 2, 4)

    # Plot the real trajectory
    ax_trajectory.plot(kitti_positions[:, 0], kitti_positions[:, 2], label=f'KITTI seq{data_track} Ground Truth')

    positions = []
    # Plot each trajectory in the list
    for trajectory_file, label in zip(trajectory_files, labels):
        trajectory = np.load(trajectory_file)
        position = np.array([pose[:3, 3] for pose in trajectory])
        ax_trajectory.plot(position[:, 0], position[:, 2], label=label, linestyle='--')
        positions.append(position)
    positions = np.array(positions)

    ax_trajectory.set_title('Trajectory Comparison')
    ax_trajectory.set_xlabel('X position')
    ax_trajectory.set_ylabel('Z position')
    ax_trajectory.legend()
    ax_trajectory.grid(True)

    # Compute and plot velocities
    velocities = np.diff(positions, axis=1)
    kitti_velocity = np.diff(kitti_positions, axis=0)

    # Plot X velocity
    ax_x_velocity.plot(kitti_velocity[:, 0], label=f'Vx GT')
    for velocity, label in zip(velocities, labels):
        ax_x_velocity.plot(velocity[:, 0], label=label, linestyle='--')
    ax_x_velocity.set_title('X Velocities Comparison')
    ax_x_velocity.set_xlabel('Time')
    ax_x_velocity.set_ylabel('X Velocity')
    # ax_x_velocity.legend()
    ax_x_velocity.grid(True)

    # Plot Z velocity
    ax_z_velocity.plot(kitti_velocity[:, 2], label=f'Vz GT')
    for velocity, label in zip(velocities, labels):
        ax_z_velocity.plot(velocity[:, 2], label=label, linestyle='--')
    ax_z_velocity.set_title('Z Velocities Comparison')
    ax_z_velocity.set_xlabel('Time')
    ax_z_velocity.set_ylabel('Z Velocity')
    # ax_z_velocity.legend()
    ax_z_velocity.grid(True)

    plt.tight_layout()
    plt.show()    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot multiple trajectories against a real trajectory.")
    parser.add_argument("--data_track", help="Data track identifier for the real trajectory.")
    parser.add_argument("--trajectory_files", nargs='+', help="File paths to the trajectories to plot.", required=True)
    parser.add_argument("--labels", nargs='+', help="Labels for each trajectory.", required=True)

    args = parser.parse_args()

    real_trajectory_path = f'{DATASET_PATH}/poses/{args.data_track}.txt'
    plot_trajectories(real_trajectory_path, args.trajectory_files, args.labels, args.data_track)
