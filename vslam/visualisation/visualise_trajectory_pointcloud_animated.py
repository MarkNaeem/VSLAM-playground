import pickle
import numpy as np
import open3d as o3d
from vslam.utils import load_poses
from vslam.definitions import *
import time
from tqdm import tqdm

DATA_TRACK = '02'

TRAJECTORY_PATH    = 'trajectories/lidar_depth_trajectory.npy'
POINTCLOUD_PATH    = 'pointclouds/lidar_depth_pointclouds.pkl'
RENDER_CONFIG_PATH = 'configs/view_config.json'

# Load the saved data
camera_trajectory = np.load(TRAJECTORY_PATH)
with open(POINTCLOUD_PATH, 'rb') as file:
    combined_data = pickle.load(file)

NUM_ITERATIONS = len(combined_data)
print("iterations number:",NUM_ITERATIONS)

real_trajectory = load_poses(f'{DATASET_PATH}/poses/{DATA_TRACK}.txt')


# Create and add point clouds to the visualiser
transformed_points = np.empty((0,3))
points_colors = np.empty((0,3))
# TODO: consider handling the case when no colors saved
for pose, combined_data in zip(camera_trajectory[1:], combined_data):  # Skip the first identity pose
    pts3d  = combined_data[:, :3]  # First 3 columns for points
    colors = combined_data[:, 3:6] / 255  # Last 3 columns for colors
    # Apply transformation: rotate and translate the points
    rotated_pts = pts3d @ pose[:3, :3].T  # Apply rotation
    transformed_pts = rotated_pts + pose[:3, 3]  # Apply translation 
    transformed_points = np.vstack([transformed_points,transformed_pts])   
    points_colors = np.vstack([points_colors,transformed_pts])   
print(transformed_points.shape)

camera_lines = [[i, i + 1] for i in range(NUM_ITERATIONS)]
real_lines = [[i, i + 1] for i in range(NUM_ITERATIONS)]
line_color = [[1,0,0] for _ in range(NUM_ITERATIONS)]


# pcd = o3d.geometry.PointCloud()
camera_line_set = o3d.geometry.LineSet()
real_line_set   = o3d.geometry.LineSet()

# Initialise Visualiser
vis = o3d.visualization.Visualizer()
vis.create_window()

# Initialise Visualiser
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(camera_line_set)
vis.add_geometry(real_line_set)
# vis.add_geometry(pcd)

# Visualisation loop
for k in tqdm(range(NUM_ITERATIONS)):
    try:
        # pcd = o3d.geometry.PointCloud()
        camera_line_set = o3d.geometry.LineSet()
        real_line_set   = o3d.geometry.LineSet()

        # Create a line set for the camera path
        camera_points_o3d = o3d.utility.Vector3dVector(camera_trajectory[:k, :3, 3])
        camera_lines_o3d  = o3d.utility.Vector2iVector(camera_lines[:k])

        # Create a line set for the real trajectory
        real_points_o3d = o3d.utility.Vector3dVector(real_trajectory[:k, :3, 3])
        real_lines_o3d  = o3d.utility.Vector2iVector(real_lines[:k])
        real_line_set.colors = o3d.utility.Vector3dVector(line_color[:k])

        camera_line_set.lines = camera_lines_o3d
        camera_line_set.points = camera_points_o3d
        real_line_set.lines = real_lines_o3d
        real_line_set.points = real_points_o3d

        # pcd.points = o3d.utility.Vector3dVector(transformed_points[:k])
        # pcd.colors = o3d.utility.Vector3dVector(points_colors[:k])

        vis.update_geometry(camera_line_set)
        vis.update_geometry(real_line_set)
        vis.poll_events()
        vis.update_renderer()

        time.sleep(1)
        
    except KeyboardInterrupt:
        print("finished")
        break

vis.destroy_window()
