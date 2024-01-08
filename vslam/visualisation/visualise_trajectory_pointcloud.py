import pickle
import numpy as np
import open3d as o3d
from vslam.utils import load_poses
from vslam.definitions import *

DATA_TRACK = '02'

TRAJECTORY_PATH    = 'trajectories/lidar_depth_trajectory.npy'
POINTCLOUD_PATH    = 'pointclouds/lidar_depth_pointclouds.pkl'
RENDER_CONFIG_PATH = 'configs/view_config.json'

# Load the saved data
camera_trajectory = np.load(TRAJECTORY_PATH)
with open(POINTCLOUD_PATH, 'rb') as file:
    combined_data = pickle.load(file)

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

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(transformed_points)
pcd.colors = o3d.utility.Vector3dVector(points_colors)

# Create a line set for the camera path
camera_lines = [[i, i + 1] for i in range(len(camera_trajectory) - 1)]
camera_line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(camera_trajectory[:, :3, 3]),
    lines=o3d.utility.Vector2iVector(camera_lines)
)

# Create a line set for the real trajectory
real_lines = [[i, i + 1] for i in range(len(real_trajectory) - 1)]
line_color = [[1,0,0] for _ in range(len(real_trajectory) - 1)]
real_line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(real_trajectory[:, :3, 3]),
    lines=o3d.utility.Vector2iVector(real_lines),
)
real_line_set.colors = o3d.utility.Vector3dVector(line_color)

# Initialise Visualiser
vis = o3d.visualization.Visualizer()
vis.create_window()

# apply visulisation configs
# vis.get_render_option().load_from_json(RENDER_CONFIG_PATH)

# Initialise Visualiser
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(camera_line_set)
vis.add_geometry(real_line_set)
vis.add_geometry(pcd)

# Visualisation loop
while True:
    try:
        vis.poll_events()
        vis.update_renderer()
    except KeyboardInterrupt:
        print("finished")
        break

vis.destroy_window()
