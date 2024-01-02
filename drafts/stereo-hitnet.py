import cv2
import numpy as np
import torch
from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import frame2tensor

from vslam.utils import *
from vslam.definitions import *

DATA_TRACK = '02'
MAX_IMG_HEIGHT = 256
MAX_DEPTH = 50
TRAJECTORY_PATH = 'trajectories/stereo_trajectory.npy'

num_images = len(os.listdir(f'{DATASET_PATH}/sequences/{DATA_TRACK}/image_2'))

# read camera intrinsics K, base line
calib_file_path = f'{DATASET_PATH}/sequences/{DATA_TRACK}/calib.txt'
K = read_kitti_calibration(calib_file_path)
K_inv = np.linalg.inv(K)
print("Intrinsic Matrix (K):\n", K)
print()
fx,bl = get_fx_baseline(f'{DATASET_PATH}/sequences/{DATA_TRACK}/calib.txt')
print("fx, bl:", fx, bl)

# Load real trajectory
real_trajectory = load_poses(f'{DATASET_PATH}/poses/{DATA_TRACK}.txt')

torch.set_grad_enabled(False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Running on device "{}"'.format(device))

# Configuration for the matching model
config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1
    },
    'superglue': {
        'weights': 'outdoor',  # or 'indoor'
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}
matching = Matching(config).eval().to(device)


depth_model = PredictModel(model="HITNet_SF", ckpt="TinyHITNet/ckpt/hitnet_sf_finalpass.ckpt")
ckpt = torch.load('TinyHITNet/ckpt/hitnet_sf_finalpass.ckpt',map_location=torch.device(device))
if "state_dict" in ckpt:
    depth_model.load_state_dict(ckpt["state_dict"])
else:
    depth_model.model.load_state_dict(ckpt)
if device == 'cuda':
    _ = depth_model.cuda()

num_images = len(os.listdir(f'{DATASET_PATH}/sequences/{DATA_TRACK}/image_2'))

# Initialize variables for tracking position and orientation
try:
    trajectory = np.load(TRAJECTORY_PATH)
    current_pose = trajectory[-1]
except:
    print("no trajectory saved found.. starting new")
    current_pose = np.eye(4)
    trajectory = [current_pose]

prev_image_path = f'{DATASET_PATH}/sequences/{DATA_TRACK}/image_2/{len(trajectory)-1:06d}.png'
color_prev_frame = cv2.imread(prev_image_path)
# color_prev_frame = scale_image(color_prev_frame,max_height=MAX_IMG_HEIGHT)
prev_frame = cv2.cvtColor(color_prev_frame, cv2.COLOR_BGR2GRAY)

prev_frame_tensor = frame2tensor(prev_frame, device)

# Process each image with the superpoint model
prev_image_data = matching.superpoint({'image': prev_frame_tensor})
prev_image_data = {k + '0': prev_image_data[k] for k in ['keypoints', 'scores', 'descriptors']}
prev_image_data['image0'] = prev_frame_tensor


###########
#  LOOP   #
###########


# Loop through the image sequence
if type(trajectory)!=list: trajectory = trajectory.tolist()
    
for i in tqdm(range(len(trajectory),num_images - 1)):
  try:
    curr_image_path = f'{DATASET_PATH}/sequences/{DATA_TRACK}/image_2/{i:06d}.png'

    # Load the previous and current images
    color_curr_frame = cv2.imread(curr_image_path)
    # color_curr_frame = scale_image(color_curr_frame,max_height=MAX_IMG_HEIGHT)
    curr_frame = cv2.cvtColor(color_curr_frame, cv2.COLOR_BGR2GRAY)

    right_frame = cv2.imread(f'{DATASET_PATH}/sequences/{DATA_TRACK}/image_3/{i:06d}.png')
    # right_frame = scale_image(right_frame,max_height=MAX_IMG_HEIGHT)
    
    # depth = compute_depth_map(color_curr_frame, right_frame,fx,bl*1e-3)

    pred = predict(depth_model, color_curr_frame, right_frame)
    disp = torch.clip(pred["disp"] / 192 * 255, 0, 255).long().cpu()[0,0]
    depth = disparity_to_depth(disp,fx,bl*1e-3).numpy()
    depth[depth>60] = 0
    
    # Convert images to tensors
    curr_frame_tensor = frame2tensor(curr_frame, device)

    curr_image_data = matching.superpoint({'image': curr_frame_tensor})
    curr_image_data = {k + '1': curr_image_data[k] for k in ['keypoints', 'scores', 'descriptors']}
    curr_image_data['image1'] = curr_frame_tensor

    # Perform matching
    pred = matching({**prev_image_data, **curr_image_data})

    kpts0 = prev_image_data['keypoints0'][0].cpu().numpy()
    kpts1 = curr_image_data['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()
    valid = matches > -1
    mkpts0 = kpts0[valid].astype(float)
    mkpts1 = kpts1[matches[valid]].astype(int)

    pts3d, deleted = keypoints_to_3d(mkpts1,depth,K,MAX_DEPTH)
    mkpts0 = np.delete(mkpts0,deleted,0)
    
    _, rvec, t, _ = cv2.solvePnPRansac(pts3d,mkpts0,K,None)
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
    
    prev_image_data = curr_image_data
    prev_image_data = {k[:-1] + '0': prev_image_data[k] for k in ['keypoints1', 'scores1', 'descriptors1']}
    prev_image_data['image0'] = curr_frame_tensor    
    
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
your_positions = np.array(your_positions)*2e1
# your_positions[:,-1]*=-1

# Plotting
plot_length = len(your_positions)
plt.figure(figsize=(10, 6))
plt.plot(kitti_positions[:plot_length, 0], kitti_positions[:plot_length, 2], label='KITTI seq02 Ground Truth')
plt.plot(your_positions[:plot_length, 0],  your_positions[:plot_length, 2], label='SuperPoint Monocular Odometry', linestyle='--')
plt.title('Trajectory Comparison')
plt.xlabel('X position')
plt.ylabel('Z position')
plt.legend()
plt.grid(True)
plt.show()