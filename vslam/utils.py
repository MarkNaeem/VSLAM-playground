import os
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def decomposeEssentialMatrix(E, pts0, pts1, K):
   # Decompose the Essential Matrix to get rotation and translation
    _, R, t, _ = cv2.recoverPose(E, pts0, pts1, K)

    # Update position and orientation
    # Create a 4x4 transformation matrix from R and t
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R  # Assuming R is a 3x3 matrix
    transformation_matrix[:3, 3] = t.reshape(-1)  # Assuming t is a 3x1 vector
    return transformation_matrix


def read_kitti_calibration(file_path):
    """
    Reads the KITTI calibration file and extracts the left camera camera intrinsic matrix.

    Args:
    file_path (str): The path to the calibration file.

    Returns:
    numpy.ndarray: The intrinsic matrix of the camera.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extracting the line that contains the intrinsic parameters.
    for line in lines:
        if line.startswith('P0'):
            intrinsic_line = line.split()[1:]

    # Convert to float and reshape into a 3x4 matrix
    intrinsic_matrix_3x4 = np.array([float(value) for value in intrinsic_line]).reshape((3, 4))

    # The intrinsic matrix K is the 3x3 upper-left part of this matrix
    K = intrinsic_matrix_3x4[:3, :3]

    return K


def plot_camera_movement(R, t):
    # Create a figure for 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Origin of the first camera frame
    ax.scatter(0, 0, 0, c='r', marker='o')

    # Origin of the second camera frame
    cam_pos = -R.T @ t
    ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], c='b', marker='o')

    # Create coordinate axes for the first camera frame
    ax.quiver(0, 0, 0, 1, 0, 0, length=0.25, color='r')
    ax.quiver(0, 0, 0, 0, 1, 0, length=0.25, color='g')
    ax.quiver(0, 0, 0, 0, 0, 1, length=0.25, color='b')

    # Create coordinate axes for the second camera frame
    cam_axes = R.T @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], *cam_axes[:, 0], length=0.25, color='r')
    ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], *cam_axes[:, 1], length=0.25, color='g')
    ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], *cam_axes[:, 2], length=0.25, color='b')

    # Set plot limits and labels
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    print("showing the figure")
    plt.show()


def rotation_matrix_to_euler_angles(R):
    # Assuming the order of angles is ZYX
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


def plot_orientation_arrow(ax, position, euler_angle, length=0.1):
    # Calculate direction vector based on Euler angles
    # Assuming the first angle (euler_angle[0]) is the heading
    dx = length * np.cos(euler_angle[0])
    dz = length * np.sin(euler_angle[0])

    # Plot an arrow to represent orientation
    ax.arrow(position[0], position[2], dx, dz, head_width=0.05, head_length=0.1, fc='blue', ec='blue')

    
def get_fx_baseline(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()        

    # Extract the P0 and P1 matrices
    P0 = [float(val) for val in lines[0].split()[1:]]
    P1 = [float(val) for val in lines[1].split()[1:]]

    # Focal length is the first element in the P0 matrix
    focal_length = P0[0]

    # Baseline is the absolute difference of the fourth element of P0 and P1 matrices
    baseline = abs(P1[3] - P0[3])

    return focal_length, baseline


def compute_depth_map(left_image, right_image, focal_length, baseline):
    window_size = 5
    min_disp = 16
    num_disp = 112 - min_disp
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=window_size,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32,
                                   disp12MaxDiff=1,
                                   P1=8*3*window_size**2,
                                   P2=32*3*window_size**2)
    disparity =  stereo.compute(left_image, right_image)
    depth_map = focal_length * baseline / (disparity + 1e-9)
    return depth_map


def convert_2d_to_3d(mkpts2d, depth_map, K_inv):
    """SLOW way to get 3d points from 2D featuers (deprecated)
    Convert 2D points to 3D using the depth map and camera intrinsic matrix"""
    pts_3d = []
    for pt2d in mkpts2d:
        x, y = pt2d
        Z = depth_map[int(y), int(x)]
        pt3d = K_inv @ np.array([x * Z, y * Z, Z])
        pts_3d.append(pt3d)
    return np.array(pts_3d)    


def load_poses(file_path):
    poses = []
    with open(file_path, 'r') as file:
        for line in file:
            pose = np.fromstring(line, dtype=float, sep=' ')
            pose = pose.reshape((3, 4))  # Reshape to 3x4 matrix
            pose = np.vstack((pose, [0, 0, 0, 1]))  # Convert to 4x4 matrix
            poses.append(pose)
    return np.asarray(poses)


def scale_image(image, max_width=None, max_height=None, scale_factor=None):
    height, width = image.shape[:2]

    # Compute the scaling factor
    if scale_factor is not None:
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
    else:
        if max_width is not None and width > max_width:
            aspect_ratio = height / width
            new_width = max_width
            new_height = int(max_width * aspect_ratio)
        elif max_height is not None and height > max_height:
            aspect_ratio = width / height
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        else:
            return image  # No resizing needed

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image


def draw_flow(image, matches, plot=True, file_name='plot.png', folder='.'):
    # Ensure the image is in the correct format
    if len(image.shape) == 2:  # grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Draw lines and points
    for match in matches:
        old_point = (int(match[0][0]), int(match[0][1]))
        new_point = (int(match[1][0]), int(match[1][1]))
        cv2.line(image, old_point, new_point, (0, 255, 0), 1)
        cv2.circle(image, new_point, 1, (0, 0, 255), -1)
        cv2.circle(image, old_point, 1, (255, 0, 0), -1)
    if plot:
        # Display the image
        plt.figure(figsize=(10, 5))
        plt.imshow( image )
        plt.show()
    else:
        save_path = os.path.join(folder, file_name)
        cv2.imwrite(save_path, image)

    
def create_video(image_folder, output_video='trajectory_video.avi', fps=10):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Ensure the frames are in the correct order

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

    for image in tqdm(images):
        video.write(cv2.imread(os.path.join(image_folder, image)))

    #cv2.destroyAllWindows()
    video.release()
    

def disparity_to_depth(disparity, focal_length, baseline):
    # Avoid division by zero
    disparity[disparity == 0] = 1e-3
    # Convert disparity to depth
    depth = (focal_length * baseline) / disparity
    return depth

    
    
def keypoints_to_3d(keypoints, depth_image, intrinsic_matrix, max_depth=None):
    """
    Convert keypoints from a depth image to 3D coordinates.
    :param keypoints: List of (u, v) tuples of keypoint pixel coordinates.
    :param depth_image: 2D array with depth values.
    :param intrinsic_matrix: 3x3 intrinsic camera matrix.
    :return: List of (X, Y, Z) 3D coordinates.
    """
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    points_3d = []
    deleted = []
    for i,(u, v) in enumerate(keypoints):
        Z = depth_image[v, u]  # Assuming depth image has depth in meters
        if (not max_depth is None and Z >= max_depth) or Z<=0:
            deleted.append(i)
            continue # ignore any point that may have large depth value ( can throw calcuations off )
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        points_3d.append((X, Y, Z))

    return np.asarray(points_3d), deleted    


def densify_depth_image_fast(depth_image, method='nearest'):
    """
    Densify a sparse depth image using fast interpolation.

    :param depth_image: Sparse depth image
    :param method: Interpolation method ('nearest', 'linear', 'cubic')
    :return: Densified depth image
    """
    # Get the image dimensions
    height, width = depth_image.shape

    # Create a meshgrid of pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Mask for valid (non-zero) depth points
    valid_mask = depth_image > 0
    valid_mask[:150,:] = True

    # Coordinates and depth values of valid points
    valid_points = np.array([x[valid_mask], y[valid_mask]]).T
    valid_depths = depth_image[valid_mask]

    # Grid coordinates for interpolation
    grid_x, grid_y = np.mgrid[0:height, 0:width]

    # Interpolate using griddata
    densified_depth = griddata(valid_points, valid_depths, (grid_y, grid_x), method=method, fill_value=0)

    return densified_depth


def read_velodyne_bin(path):
    """
    Reads a .bin file from KITTI dataset containing Velodyne point cloud data.
    """
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return points


def visualize_lidar_data(points):
    """
    Visualizes the LiDAR point cloud data with open3d.
    """
    # Open3D does not use the intensity and ring index, so we slice only the XYZ values
    xyz = points[:, :3]

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])


def visualize_lidar_sequence(sequence_folder):
    """
    Visualizes a sequence of LiDAR point cloud data from the KITTI dataset with open3d.
    """
    # Initialize the Open3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for file in sorted(os.listdir(sequence_folder)):
        if file.endswith('.bin'):
            # Read the point cloud data
            file_path = os.path.join(sequence_folder, file)
            points = read_velodyne_bin(file_path)

            # Open3D does not use the intensity and ring index, so we slice only the XYZ values
            xyz = points[:, :3]

            # Create Open3D point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)

            # Visualize the point cloud
            vis.add_geometry(pcd)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            vis.remove_geometry(pcd)

    vis.destroy_window()


def point_cloud_to_depth_image(points, img_width=1242, img_height=375, fov_up=3.0, fov_down=-25.0):
    """
    Converts a point cloud to a depth image.
    """
    # Filter out points with 0 depth value
    valid_indices = points[:, 0] != 0
    points = points[valid_indices]

    # Calculate the depth value
    depth = np.linalg.norm(points[:, :3], axis=1)

    # Calculate the horizontal and vertical angles
    horiz_angle = np.arctan2(points[:, 1], points[:, 0]) * 180 / np.pi
    vert_angle = np.arcsin(points[:, 2] / depth) * 180 / np.pi

    # Calculate the pixel coordinates
    horiz_pixel = (horiz_angle + 180) / 360 * img_width
    vert_pixel = (vert_angle + fov_up) / (fov_up - fov_down) * img_height

    # Initialize depth image
    depth_image = np.zeros((img_height, img_width))

    # Assign depth value to the image pixels
    for i in range(len(points)):
        col = int(np.round(horiz_pixel[i]))
        row = int(np.round(vert_pixel[i]))
        if 0 <= row < img_height and 0 <= col < img_width:
            depth_image[row, col] = depth[i]

    # Normalize the depth image
    depth_image = depth_image / np.max(depth_image) * 255
    depth_image = depth_image.astype(np.uint8)

    return depth_image


def visualize_depth_sequence(sequence_folder):
    """
    Visualizes a sequence of lidar scans (.bin files) from the KITTI dataset as depth images.
    """
    for file in sorted(os.listdir(sequence_folder)):
        if file.endswith('.bin'):
            # Read the point cloud data
            file_path = os.path.join(sequence_folder, file)
            points = read_velodyne_bin(file_path)

            # Convert to depth image
            depth_image = point_cloud_to_depth_image(points)

            # Display the depth image
            cv2.imshow('Depth Image', depth_image)
            cv2.waitKey(1)  # Display each frame for 1ms

    cv2.destroyAllWindows()


def create_depth_image(image_points, depths, image_width, image_height):
    """
    Create a depth image from 2D image points and their respective depth values.
    returns the depth image and a normalised version of it (mainly for display)
    """
    # Initialize an empty image
    depth_image = np.zeros((image_height, image_width), dtype=np.float32)

    # Round the image points to the nearest integer to get pixel coordinates
    image_points = np.rint(image_points).astype(int)

    # Populate the depth image with depth values
    for point, depth in zip(image_points, depths):
        x, y = point
        if 0 <= x < image_width and 0 <= y < image_height:
            depth_image[y, x] = depth

    # Normalize the depth image for visualization
    depth_image_normalised = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_image_normalised = np.uint8(depth_image_normalised)

    return depth_image, depth_image_normalised


def load_calibration_data(calib_file):
    """
    Load the intrinsic and extrinsic calibration data from a KITTI calibration file.
    """
    with open(calib_file, 'r') as file:
        lines = file.readlines()

    # Extract the calibration matrices
    P0_line = next(line for line in lines if line.startswith('P0'))
    Tr_line = next(line for line in lines if line.startswith('Tr'))

    # Parse the P0 (intrinsic parameters for left color camera) and Tr (extrinsic parameters) matrices
    P0 = np.array([float(value) for value in P0_line.split()[1:13]]).reshape(3, 4)
    Tr = np.array([float(value) for value in Tr_line.split()[1:13]]).reshape(3, 4)

    # Convert Tr into a 4x4 matrix for homogeneous coordinates
    Tr_homogeneous = np.eye(4)
    Tr_homogeneous[:3, :4] = Tr

    # The intrinsic matrix is the first 3x3 part of P2
    intrinsic_calib = P0[:, :3]

    return intrinsic_calib, Tr_homogeneous


def project_lidar_to_camera(lidar_points, extrinsic_calib, intrinsic_calib, image_width, image_height):
    """Projects 3D lidar points onto an image plane. Returns 2D image points (on the image plane) and their depth values"""
    # Transform LiDAR points to camera coordinate system
    lidar_points_homogeneous = np.hstack((lidar_points[:, :3], np.ones((lidar_points.shape[0], 1))))
    lidar_points_camera_homogeneous = np.dot(extrinsic_calib, lidar_points_homogeneous.T).T

    # Convert from homogeneous 4D to 3D coordinates - Normalise the points
    lidar_points_camera = lidar_points_camera_homogeneous[:, :3] / lidar_points_camera_homogeneous[:, 3:4]

    # Remove points behind the camera
    lidar_points_camera = lidar_points_camera[lidar_points_camera[:, 2] > 0]

    # Project 3D points onto 2D image plane
    image_points_homogeneous = np.dot(intrinsic_calib, lidar_points_camera.T).T

    # Convert from 3D to 2D coordinates - Normalise the points
    image_points = image_points_homogeneous[:, :2] / image_points_homogeneous[:, 2:3]

    # Filter points outside the camera FoV
    valid_indices = (image_points[:, 0] >= 0) & (image_points[:, 0] < image_width) & \
                    (image_points[:, 1] >= 0) & (image_points[:, 1] < image_height)
    image_points = image_points[valid_indices]
    depths = lidar_points_camera[valid_indices, 2]

    return image_points, depths


def run_icp(source_points, target_points, threshold=1.0, trans_init=np.identity(4)):
    """
    Perform ICP alignment on two sets of 3D points.

    :param source_points: Source point cloud (Nx3 numpy array).
    :param target_points: Target point cloud (Nx3 numpy array).
    :param threshold: Distance threshold for point correspondences.
    :param trans_init: Initial transformation guess (4x4 numpy array).
    :return: Transformed source point cloud, transformation matrix.
    """
    # Convert numpy arrays to Open3D point clouds
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)
    target.points = o3d.utility.Vector3dVector(target_points)

    # Run ICP
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )

    # Transform the source point cloud using the obtained transformation
    # source.transform(reg_p2p.transformation)

    # return source, reg_p2p.transformation
    return reg_p2p.transformation