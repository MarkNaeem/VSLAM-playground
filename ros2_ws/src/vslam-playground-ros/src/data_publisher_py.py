import os
import cv2
from tqdm import tqdm
from queue import Queue
from threading import Thread

from vslam.utils import *
from vslam.definitions import *

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

import tf2_ros
import tf_transformations

from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo, PointCloud2


class DataPublisher(Node):
    def __init__(self):
        super().__init__('data_publisher')
        self.cv_bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, 'py_image_color', 10)
        self.depth_pub = self.create_publisher(Image, 'py_image_depth', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, 'py_lidar_points', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, 'py_camera_info', 10)

        self.data_track = '02'
        self.dataset_path = DATASET_PATH
        self.num_images = len(os.listdir(f'{self.dataset_path}/sequences/{self.data_track}/image_2'))

        self.intrinsic_matrix, self.extrinsic_matrix = load_calibration_data(f'{self.dataset_path}/sequences/{self.data_track}/calib.txt')

        self.image_queue = Queue()
        self.depth_queue = Queue()
        self.pointcloud_queue = Queue()

        self.publish_static_transform()

        image_loader_thread = Thread(target=self.image_loader)
        depth_loader_thread = Thread(target=self.depth_loader)

        image_loader_thread.start()
        depth_loader_thread.start()
        
        self.publish_data()

    def image_loader(self):
        for i in range(self.num_images):
            color_image_path = f'{self.dataset_path}/sequences/{self.data_track}/image_2/{i:06d}.png'
            color_image = cv2.imread(color_image_path)
            self.image_queue.put(color_image)


    def depth_loader(self):
        for i in range(self.num_images):
            lidar_path = f'{self.dataset_path}/sequences/{self.data_track}/velodyne/{i:06d}.bin'
            lidar_points = read_velodyne_bin(lidar_path)
            image_points, depths = project_lidar_to_camera(lidar_points, self.extrinsic_matrix, self.intrinsic_matrix, self.intrinsic_matrix[0, 2] * 2, self.intrinsic_matrix[1, 2] * 2)
            lidar_depth_image, _ = create_depth_image(image_points, depths, int(self.intrinsic_matrix[0, 2] * 2), int(self.intrinsic_matrix[1, 2] * 2))
            # depth = densify_depth_image_fast(lidar_depth_image)
            point_cloud = self.convert_lidar_to_PointCloud2(lidar_points)
            
            self.depth_queue.put(lidar_depth_image)
            self.pointcloud_queue.put(point_cloud)


    def publish_data(self):
        for _ in tqdm(range(self.num_images)):
            color_image = self.image_queue.get()
            depth = self.depth_queue.get()
            point_cloud = self.pointcloud_queue.get()

            self.publish_image(color_image, self.image_pub)
            self.publish_image(depth, self.depth_pub, encoding='32FC1')
            self.pointcloud_pub.publish(point_cloud)

            camera_info_msg = self.create_camera_info_msg()
            self.camera_info_pub.publish(camera_info_msg)


    def publish_image(self, cv_image, publisher, encoding='bgr8'):
        ros_image = self.cv_bridge.cv2_to_imgmsg(cv_image, encoding)
        publisher.publish(ros_image)


    def convert_lidar_to_PointCloud2(self, lidar_points):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'lidar_link'
        cloud = point_cloud2.create_cloud_xyz32(header, lidar_points[:, :3].tolist())
        return cloud


    def create_camera_info_msg(self):
        camera_info_msg = CameraInfo()
        camera_info_msg.header.frame_id = 'camera_link'
        camera_info_msg.height = int(self.intrinsic_matrix[1, 1])
        camera_info_msg.width = int(self.intrinsic_matrix[0, 0])
        camera_info_msg.k = self.intrinsic_matrix.flatten().tolist()
        return camera_info_msg


    def publish_static_transform(self):
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'camera_link'
        t.child_frame_id = 'lidar_link'
        # Set the translation from the extrinsic matrix
        t.transform.translation.x = self.extrinsic_matrix[0, 3]
        t.transform.translation.y = self.extrinsic_matrix[1, 3]
        t.transform.translation.z = self.extrinsic_matrix[2, 3]
        # Extract the rotation matrix and convert it to a quaternion
        quaternion = tf_transformations.quaternion_from_matrix(self.extrinsic_matrix)
        # Set the rotation in the transform
        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]
        # Send the transform
        self.tf_broadcaster.sendTransform(t)

        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'camera_link'
        # Set the translation from the extrinsic matrix
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        # Extract the rotation matrix and convert it to a quaternion
        quaternion = tf_transformations.quaternion_from_matrix(self.extrinsic_matrix.T)
        # Set the rotation in the transform
        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]
        # Send the transform
        self.tf_broadcaster.sendTransform(t)



def main(args=None):
    rclpy.init(args=args)
    data_publisher = DataPublisher()
    rclpy.spin(data_publisher)
    data_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
