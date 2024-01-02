import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
import cv2
import torch
import numpy as np
from cv_bridge import CvBridge
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import os

from vslam.utils import *

class VisualOdometryNode(Node):
    def __init__(self):
        super().__init__('visual_odometry_node')

        # ROS2 Subscriptions
        self.image_subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10)
        self.lidar_subscription = self.create_subscription(
            PointCloud2,
            'lidar/points',
            self.lidar_callback,
            10)

        # ROS2 Publisher
        self.pose_publisher = self.create_publisher(PoseStamped, 'robot/pose', 10)

        # Initialization of variables
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

        # Load calibration data and set up other parameters
        self.initialize_odometry()


    def initialize_odometry(self):
        # Read the calibration file path from a ROS parameter
        calib_file_path = self.get_parameter('calib_file_path').get_parameter_value().string_value

        if not calib_file_path:
            self.get_logger().error('Calibration file path parameter not set. Shutting down.')
            rclpy.shutdown()
            return


        self.intrinsic_matrix, self.extrinsic_matrix = load_calibration_data(calib_file_path)
        # Load other odometry initialization as required
        pass    
    

    def image_callback(self, msg):
        # Process the image
        current_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # Add your image processing code here


    def lidar_callback(self, msg):
        # Process LiDAR data
        # Convert PointCloud2 to appropriate format and process
        pass


    def publish_pose(self, pose):
        # Publish the pose as a PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "odom"
        # Fill pose_msg.pose with pose data
        self.pose_publisher.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)

    visual_odometry_node = VisualOdometryNode()

    # Declare the calibration file path parameter
    visual_odometry_node.declare_parameter('calib_file_path', '')

    rclpy.spin(visual_odometry_node)

    visual_odometry_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
