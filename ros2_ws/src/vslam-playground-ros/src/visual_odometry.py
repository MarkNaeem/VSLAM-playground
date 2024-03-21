#!/usr/bin/env python3

import cv2
import torch
import numpy as np
from queue import Queue
from sensor_msgs.msg import CameraInfo

import rclpy
import tf2_ros
import tf_transformations
from rclpy.node import Node
from nav_msgs.msg import Path
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped, PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber

from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import frame2tensor

from vslam.utils import keypoints_to_3d
from vslam.definitions import *


DATA_TRACK = '02'
TRAJECTORY_PATH = 'trajectories/ros2_trajectory.npy'


def transform_to_matrix(transform):
    """
    Convert a geometry_msgs Transform to a 4x4 homogeneous transformation matrix.
    """
    translation = [transform.translation.x, transform.translation.y, transform.translation.z]
    rotation = [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]
    
    # Create the transformation matrix
    matrix = tf_transformations.concatenate_matrices(
        tf_transformations.quaternion_matrix(rotation),
        tf_transformations.translation_matrix(translation)
    )
    return matrix


def matrix_to_translation_rotation(matrix):
    scale, shear, angles, trans, persp = tf_transformations.decompose_matrix(matrix)
    return tf_transformations.quaternion_from_euler(*angles), trans


class VisualOdometryNode(Node):
    def __init__(self):
        super().__init__('visual_odometry_node')

        # Matching configuration and initialization
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.matching = self.initialize_matching()
        self.prev_image_data = None
        self.current_pose = np.eye(4)
        self.current_pose_camera_frame = np.eye(4)

        self.bridge = CvBridge()
        
        self._counter = 0
        self.MAX_DEPTH = 60
        self.FREQ = 10
        
        self.images_queue = Queue(maxsize=10)

        self.PROCESS_FREQ = 10.0
        self.PROCESS_TIME = 1 / self.PROCESS_FREQ

        self.get_logger().info("waiting for camera to base link transform...")
        buffer = tf2_ros.Buffer()
        listener = tf2_ros.transform_listener.TransformListener(buffer, self)
        transform = None
        for _ in range(10):
            if buffer.can_transform('base_link', 'camera_link', rclpy.time.Time()):
                transform = buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time())
                break
            rclpy.spin_once(self, timeout_sec=2.0)  # Wait a bit for the transform to become available
        if transform is None:
            raise RuntimeError("Failed to find transform from camera_link to base_link")
        else:
            self.get_logger().info("Transform found")
        self.camera_transform = transform_to_matrix(transform.transform)
        self.get_logger().info(self.camera_transform)
            
        # Subscriber for camera_info
        self.camera_sub = self.create_subscription(CameraInfo, 'camera_info', self.camera_info_callback, 10)
        
        # Placeholder for the intrinsic matrix
        self.intrinsic_matrix = None

        # Initialize subscribers using message_filters for synchronization
        self.depth_sub = Subscriber(self, Image, 'image_depth')
        self.color_sub = Subscriber(self, Image, 'image_color')

        # Synchronizer
        ats = ApproximateTimeSynchronizer([self.depth_sub, self.color_sub], queue_size=5, slop=0.1)
        ats.registerCallback(self.sync_callback)

        # TF Broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Path Publisher
        self.path_publisher = self.create_publisher(Path, 'vo_path', 10)
        self.trajectory = Path()
        self.trajectory.header.frame_id = "odom"

        self.timer = self.create_timer(self.PROCESS_TIME, self.process_loop)


    def initialize_matching(self):
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': -1
            },
            'superglue': {
                'weights': 'outdoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
        matching = Matching(config).eval().to(self.device)
        return matching
    

    def camera_info_callback(self, msg):
        if self.intrinsic_matrix is None:
            K = np.array(msg.k).reshape(3, 3)
            self.intrinsic_matrix = K
            self.get_logger().info(f"Updated intrinsic matrix: \n{self.intrinsic_matrix}")
            self.destroy_subscription(self.camera_sub)


    def sync_callback(self, depth_msg, color_msg):
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        self.images_queue.put( (depth_image, color_image) )

        self._counter+=1



    def process_loop(self):
        try:
            (depth_image, color_image) = self.images_queue.get_nowait()
        except:
            self.get_logger().warn("Queue Empty!! waiting for images..",throttle_duration_sec=1)
            return 
           
        # Process images for visual odometry
        self.process_images(depth_image, color_image)      

        # publish the current pose n the path topic and the tf tree
        self.publish_path()
        self.publish_tf()


    def process_images(self, depth_image, color_image):

        if self.intrinsic_matrix is None:
            self.get_logger().warn("waiting for camera_info topic..",throttle_duration_sec=1)
            return

        # Convert the current color image to a tensor
        curr_frame_tensor = frame2tensor(color_image, self.device)

        curr_image_data = self.matching.superpoint({'image': curr_frame_tensor})
        curr_image_data = {k + '1': curr_image_data[k] for k in ['keypoints', 'scores', 'descriptors']}
        curr_image_data['image1'] = curr_frame_tensor

        # Initialize prev_image_data if it's the first frame
        if self.prev_image_data is None:
            # Process each image with the superpoint model
            self.prev_image_data = {k[:-1] + '0': curr_image_data[k] for k in ['keypoints1', 'scores1', 'descriptors1']}
            self.prev_image_data['image0'] = curr_frame_tensor
            return

        # Perform matching
        pred = self.matching({**self.prev_image_data, **curr_image_data})
        
        # Extract matched keypoints
        kpts0 = self.prev_image_data['keypoints0'][0].cpu().numpy()
        kpts1 = curr_image_data['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        valid = matches > -1
        mkpts0 = kpts0[valid].astype(float)
        mkpts1 = kpts1[matches[valid]].astype(int)

        pts3d, deleted = keypoints_to_3d(mkpts1,depth_image,self.intrinsic_matrix,self.MAX_DEPTH)
        mkpts0 = np.delete(mkpts0,deleted,0)
        _, rvec, tvec, _ = cv2.solvePnPRansac(pts3d,mkpts0,self.intrinsic_matrix,None)
        
        # Convert rvec (rotation vector) and tvec (translation vector) to a transformation matrix
        new_pose = np.eye(4)
        R, _ = cv2.Rodrigues(rvec)
        new_pose[:3, :3] = R
        new_pose[:3, 3] = tvec.reshape(-1)

        # Update the current pose
        self.current_pose_camera_frame =  self.current_pose_camera_frame @ new_pose
        self.current_pose =  self.camera_transform @ self.current_pose_camera_frame @ self.camera_transform.T

        # Prepare for the next frame
        self.prev_image_data = curr_image_data
        self.prev_image_data = {k[:-1] + '0': self.prev_image_data[k] for k in ['keypoints1', 'scores1', 'descriptors1']}
        self.prev_image_data['image0'] = curr_frame_tensor 


    def publish_path(self):
        quaternion, translation = matrix_to_translation_rotation(self.current_pose)
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = "odom"            
        pose_stamped.pose.position.x = translation[0]
        pose_stamped.pose.position.y = translation[1]
        pose_stamped.pose.position.z = translation[2]
        pose_stamped.pose.orientation.x = quaternion[0] 
        pose_stamped.pose.orientation.y = quaternion[1]
        pose_stamped.pose.orientation.z = quaternion[2]
        pose_stamped.pose.orientation.w = quaternion[3]        
        self.trajectory.poses.append(pose_stamped)
        self.path_publisher.publish(self.trajectory)
    

    def publish_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        pose = self.current_pose
        quaternion, translation = matrix_to_translation_rotation(pose)
        t.transform.translation.x  = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        t.transform.rotation.x = quaternion[0] 
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = VisualOdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
