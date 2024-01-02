import os
import cv2
import numpy as np
from utils import *
from vslam.definitions import *


def visualize_lidar_as_camera_view(lidar_data_path, image_data_path, extrinsic_calib, intrinsic_calib, image_width, image_height):
    for lidar_file in sorted(os.listdir(lidar_data_path)):
        if lidar_file.endswith('.bin'):
            # Read LiDAR data
            lidar_points = read_velodyne_bin(os.path.join(lidar_data_path, lidar_file))

            # Project LiDAR points onto camera image plane
            image_points, depths = project_lidar_to_camera(lidar_points, extrinsic_calib, intrinsic_calib, image_width, image_height)
            print(f"(MIN,MAX):({np.min(depths):03.3f},{np.max(depths):03.3f})")

            # Create depth image
            depth_image = create_depth_image(image_points, depths, image_width, image_height)
            # Read the corresponding color image
            color_image_filename = lidar_file.replace('.bin', '.png')  # Adjust the extension if necessary
            color_image_path = os.path.join(image_data_path, color_image_filename)
            color_image = cv2.imread(color_image_path)
            if color_image is not None:
                # Overlay the depth points on the color image
                overlay_image = overlay_depth_on_color(depth_image, color_image)
                cv2.imshow('Overlay Image', overlay_image)
                cv2.waitKey(1)

    cv2.destroyAllWindows()


def overlay_depth_on_color(depth_image, color_image, depth_unicolor=True ,depth_threshold=1, colormap=cv2.COLORMAP_HOT):
    """
    Overlay depth points on the color image with conditional transparency.
    """
    if depth_unicolor:
        depth_image[depth_image >= depth_threshold] = 255


    # Scale depth image to full 8-bit range (0-255) for better color mapping
    depth_scaled = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_scaled = np.uint8(depth_scaled)

    # Apply color map to depth image
    depth_colored = cv2.applyColorMap(depth_scaled, colormap)

    # Create an alpha channel based on depth values
    alpha_channel = np.ones(depth_image.shape, dtype=np.uint8) * 255
    alpha_channel[depth_image < depth_threshold] = 0

    # Merge alpha channel with depth colored image
    depth_colored_with_alpha = cv2.merge((depth_colored, alpha_channel))

    # merge an alpha channel with color to be overlayed properly
    alpha_channel = np.ones(depth_image.shape, dtype=np.uint8) * 255
    color_image_with_alpha = cv2.merge((color_image, alpha_channel))

    # Overlay the depth image on the color image
    overlay = cv2.addWeighted(color_image_with_alpha, 1.0, depth_colored_with_alpha, 1.0, 0)

    return overlay


if __name__ == "__main__":
    seq_path = f"{DATASET_PATH}/sequences/00"

    example_img = cv2.imread(f"{seq_path}/image_2/000000.png",0)
    h,w = example_img.shape[0],example_img.shape[1]
    print("image dimensions are: ",w,h)

    # Load calibration data
    intrinsic_calib, extrinsic_calib = load_calibration_data(f"{seq_path}/calib.txt")
    print("intrinsics:\n",intrinsic_calib)
    print()
    print("extrinsics:\n",extrinsic_calib)
    print()

    # Visualize
    visualize_lidar_as_camera_view(f"{seq_path}/velodyne", f"{seq_path}/image_2", extrinsic_calib, intrinsic_calib, w, h)

    # # Visualize the sequence
    # visualize_lidar_sequence("{seq_path}/velodyne")
