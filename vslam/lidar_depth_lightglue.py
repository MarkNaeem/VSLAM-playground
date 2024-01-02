import cv2
import torch
import numpy as np
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd

from vslam.utils import *
from vslam.definitions import *

DATA_TRACK = '04'
MAX_DEPTH = 70
TRAJECTORY_PATH = 'trajectories/lidar_depth_lightglue_trajectory.npy'

num_images = len(os.listdir(f'{DATASET_PATH}/sequences/{DATA_TRACK}/image_2'))

# read camera intrinsics and extrinsics
calib_file_path = f'{DATASET_PATH}/sequences/{DATA_TRACK}/calib.txt'
intrinsic_matrix, extrinsic_matrix = load_calibration_data(calib_file_path)
print("intrinsics:\n",intrinsic_matrix)
print()
print("extrinsics:\n",extrinsic_matrix)
print()

# Load real trajectory
real_trajectory = load_poses(f'{DATASET_PATH}/poses/{DATA_TRACK}.txt')


torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
FEAT_MAX = 1024
# extractor = DISK(max_num_keypoints=FEAT_MAX).eval().to(device)  # load the extractor
extractor = SuperPoint(max_num_keypoints=FEAT_MAX).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)


# Initialize variables for tracking position and orientation
try:
    trajectory = np.load(TRAJECTORY_PATH)
    current_pose = trajectory[-1]
except:
    print("no trajectory saved found.. starting new")
    current_pose = np.eye(4)
    trajectory = [current_pose]


prev_image_path = f'{DATASET_PATH}/sequences/{DATA_TRACK}/image_2/{len(trajectory)-1:06d}.png'
prev_frame_tensor = load_image(prev_image_path)
_, image_height, image_width = prev_frame_tensor.shape

# Process each image with the superpoint model
feats0 = extractor.extract(prev_frame_tensor.to(device))


###########
#  LOOP   #
###########


# Loop through the image sequence
if type(trajectory)!=list: trajectory = trajectory.tolist()
    
for i in tqdm(range(len(trajectory),num_images - 1)):
  try:
    curr_image_path = f'{DATASET_PATH}/sequences/{DATA_TRACK}/image_2/{i:06d}.png'

    # Load the previous and current images
    curr_frame_tensor = load_image(curr_image_path)

    lidar_path = f'{DATASET_PATH}/sequences/{DATA_TRACK}/velodyne/{i:06d}.bin'
    lidar_points = read_velodyne_bin(lidar_path)
    image_points, depths = project_lidar_to_camera(lidar_points, extrinsic_matrix, intrinsic_matrix, image_width, image_height)
    lidar_depth_image, _ = create_depth_image(image_points, depths, image_width, image_height)
    depth = densify_depth_image_fast(lidar_depth_image)

    feats1 = extractor.extract(curr_frame_tensor.to(device))
    feats_temp = {k:v for k,v in feats1.items()} # because it changes later in the rbd function

    matches01 = matcher({"image0": feats0, "image1": feats1})

    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    mkpts0 = torch.round(kpts0[matches[..., 0]]).float().numpy()
    mkpts1 = torch.round(kpts1[matches[..., 1]]).int().numpy()

    pts3d, deleted = keypoints_to_3d(mkpts1,depth,intrinsic_matrix,MAX_DEPTH)
    mkpts0 = np.delete(mkpts0,deleted,0)
    
    _, rvec, t, _ = cv2.solvePnPRansac(pts3d,mkpts0,intrinsic_matrix,None)
    R = cv2.Rodrigues(rvec)[0]    
    
    # Update position and orientation
    # Create a 4x4 transformation matrix from R and t
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R  # Assuming R is a 3x3 matrix
    transformation_matrix[:3, 3] = t.reshape(-1)  # Assuming t is a 3x1 vector

    # Update the current pose
    current_pose = current_pose @ transformation_matrix

    # Add the updated pose to the trajectory
    trajectory.append(current_pose)
    
    feats0 = {k:v for k,v in feats_temp.items()}
    
  except KeyboardInterrupt:
    print("stopping...")
    break

trajectory = np.asarray(trajectory)
np.save(TRAJECTORY_PATH,trajectory,allow_pickle=True)


# Extract positions from the trajectories
kitti_positions = [pose[:3, 3] for pose in real_trajectory]  # Assuming kitti_trajectory is loaded from the dataset
your_positions = [pose[:3, 3] for pose in trajectory]

# Convert to arrays for easy plotting
kitti_positions = np.array(kitti_positions)
your_positions = np.array(your_positions)
# your_positions[:,-1]*=-1

# Plotting
plot_length = len(your_positions)
plt.figure(figsize=(10, 6))
plt.plot(kitti_positions[:plot_length, 0], kitti_positions[:plot_length, 2], label=f'KITTI seq{DATA_TRACK} Ground Truth')
plt.plot(your_positions[:plot_length, 0],  your_positions[:plot_length, 2], label='LightGlue+LiDAR Depth Odometry', linestyle='--')
plt.title('Trajectory Comparison')
plt.xlabel('X position')
plt.ylabel('Z position')
plt.legend()
plt.grid(True)
plt.show()