#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <queue>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "utils.hpp"

class DataPublisher : public rclcpp::Node {
public:
    DataPublisher(): Node("data_publisher"),
                     steady_clock(RCL_STEADY_TIME)
      { 
        std::string base_directory = "/home/marknaeem97/kitti-dataset";
        std::string data_track = "02";
        image_directory = base_directory + "/sequences/" + data_track + "/image_2";
        lidar_directory = base_directory + "/sequences/" + data_track + "/velodyne";
        calib_file_path = base_directory + "/sequences/" + data_track + "/calib.txt";

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(_publishData_millis),
            std::bind(&DataPublisher::publishData, this));
    
        _num_images = getNumberOfImages(image_directory);
        image_pub = this->create_publisher<sensor_msgs::msg::Image>("image_color", 10);
        depth_pub = this->create_publisher<sensor_msgs::msg::Image>("image_depth", 10);
        pointcloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("lidar_points", 10);
        camera_info_pub = this->create_publisher<sensor_msgs::msg::CameraInfo>("camera_info", 10);
        tf_broadcaster = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
        
        // Load calibration data
        loadCalibrationData(calib_file_path, intrinsic_matrix, extrinsic_matrix);

        // Convert intrinsic_matrix to string
        std::stringstream ss_intrinsic;
        ss_intrinsic << intrinsic_matrix;
        std::string intrinsic_str = ss_intrinsic.str();
        RCLCPP_INFO(this->get_logger(), "Intrinsic Matrix:\n%s", intrinsic_str.c_str());

        // Convert extrinsic_matrix to string
        std::stringstream ss_extrinsic;
        ss_extrinsic << extrinsic_matrix;
        std::string extrinsic_str = ss_extrinsic.str();
        RCLCPP_INFO(this->get_logger(), "Extrinsic Matrix:\n%s", extrinsic_str.c_str());

        camera_info_msg = createCameraInfoMsg();

        image_loader_thread = std::thread(&DataPublisher::imageLoader, this);
        depth_loader_thread = std::thread(&DataPublisher::depthLoader, this);

        publishStaticTransform();
    }

    ~DataPublisher()
    {
        RCLCPP_INFO(this->get_logger(), "Joining the threads...");
        if(image_loader_thread.joinable()) {
            image_loader_thread.join();
        }
        if(depth_loader_thread.joinable()) {
            depth_loader_thread.join();
        }
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_broadcaster;

    rclcpp::TimerBase::SharedPtr timer_;

    Eigen::Matrix3f intrinsic_matrix;
    Eigen::Matrix4f extrinsic_matrix;

    std::queue<cv::Mat> image_queue;
    std::queue<cv::Mat> depth_queue;
    std::queue<sensor_msgs::msg::PointCloud2> pointcloud_queue;

    rclcpp::Clock steady_clock;

    // Threads to load images and depth data
    std::thread image_loader_thread;
    std::thread depth_loader_thread;

    int _publishData_millis = 33;
    int _data_counter = 0;
    int _num_images;
    std::string image_directory;
    std::string lidar_directory;
    std::string calib_file_path;

    sensor_msgs::msg::CameraInfo camera_info_msg;

    void imageLoader()
    {
        RCLCPP_INFO(this->get_logger(),"Image loader thread started.");
        for (int i = 0; i < _num_images; ++i) {
            std::stringstream ss;
            ss << std::setw(6) << std::setfill('0') << i;
            std::string image_path = image_directory + "/" + ss.str() + ".png";
            cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
            image_queue.push(image);
        }
        RCLCPP_INFO(this->get_logger(),"Image loader thread finished.");
    }


    void depthLoader()
    {
        RCLCPP_INFO(this->get_logger(),"Depth loader thread started.");
        for (int i = 0; i < _num_images; ++i) {
            std::stringstream ss;
            ss << std::setw(6) << std::setfill('0') << i;
            std::string lidar_path = lidar_directory + "/" + ss.str() + ".bin";
            pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_data = readLidarData(lidar_path);
            sensor_msgs::msg::PointCloud2 rosCloud;
            pcl::toROSMsg(*lidar_data, rosCloud);
            rosCloud.header.frame_id = "lidar_link";
            rosCloud.header.stamp = this->get_clock()->now();
            rosCloud.is_dense = false;
            pointcloud_queue.push(rosCloud);            

            cv::Mat depth_image = projectLidarDataToDepthImageFast(lidar_data, extrinsic_matrix, intrinsic_matrix);
            // cv::Mat densified_depth_image;
            // densifyDepthImageFast(depth_image, densified_depth_image);
            depth_queue.push(depth_image);
        }
        RCLCPP_INFO(this->get_logger(),"Depth loader thread finished.");
    }


    void publishData() {
        if (_data_counter==0)
        {            
            RCLCPP_INFO_ONCE(this->get_logger(), "Starting data publishing at %f Hz...",1.0/(_publishData_millis*0.001f));
        }
        if (!image_queue.empty() && !pointcloud_queue.empty() && !depth_queue.empty()) {
            RCLCPP_INFO(this->get_logger(), "Publishing data for image %d of %d", _data_counter + 1, _num_images);
            _data_counter++;

            // color image
            cv::Mat color_image = image_queue.front();
            image_queue.pop();

            auto imgs_header = std_msgs::msg::Header();
            imgs_header.frame_id = "camera_frame";
            imgs_header.stamp = this->get_clock()->now();

            auto ros_image = cv_bridge::CvImage(imgs_header, "bgr8", color_image).toImageMsg();
            image_pub->publish(*ros_image);

            // depth image
            cv::Mat depth_image = depth_queue.front();
            depth_queue.pop();

            auto ros_depth = cv_bridge::CvImage(imgs_header, "32FC1", depth_image).toImageMsg();
            depth_pub->publish(*ros_depth);

            // point cloud
            sensor_msgs::msg::PointCloud2 point_cloud = pointcloud_queue.front();
            pointcloud_queue.pop();
            pointcloud_pub->publish(point_cloud);

            // camera info 
            camera_info_pub->publish(camera_info_msg);

        }
        else if (_data_counter >= _num_images)
        {
            RCLCPP_INFO_THROTTLE(this->get_logger(), steady_clock, 5000, "Sequence is finished. Please stop the node...");
        }
        else
        {
            RCLCPP_INFO_THROTTLE(this->get_logger(), steady_clock, 5000, "Waiting for data...");
        }
    }


    sensor_msgs::msg::CameraInfo createCameraInfoMsg() {
        sensor_msgs::msg::CameraInfo camera_info_msg;
        camera_info_msg.header.frame_id = "camera_link";
        camera_info_msg.height = static_cast<uint32_t>(intrinsic_matrix(1, 1));
        camera_info_msg.width = static_cast<uint32_t>(intrinsic_matrix(0, 0));
        for (int i = 0; i < 9; ++i) {
            camera_info_msg.k[i] = intrinsic_matrix(i / 3, i % 3);
        }
        return camera_info_msg;
    }


    void publishStaticTransform() {
        RCLCPP_INFO(this->get_logger(), "Publishing Static TF...");
 
        geometry_msgs::msg::TransformStamped camera_lidar_transform;
        camera_lidar_transform.header.stamp = this->get_clock()->now();
        camera_lidar_transform.header.frame_id = "camera_link";
        camera_lidar_transform.child_frame_id = "lidar_link";

        // Set the translation from the extrinsic matrix
        camera_lidar_transform.transform.translation.x = extrinsic_matrix(0, 3);
        camera_lidar_transform.transform.translation.y = extrinsic_matrix(1, 3);
        camera_lidar_transform.transform.translation.z = extrinsic_matrix(2, 3);

        // Extract the rotation matrix and convert it to a quaternion
        tf2::Matrix3x3 rotation_matrix(
            extrinsic_matrix(0, 0), extrinsic_matrix(0, 1), extrinsic_matrix(0, 2),
            extrinsic_matrix(1, 0), extrinsic_matrix(1, 1), extrinsic_matrix(1, 2),
            extrinsic_matrix(2, 0), extrinsic_matrix(2, 1), extrinsic_matrix(2, 2));
        tf2::Quaternion quaternion;
        rotation_matrix.getRotation(quaternion);

        // Set the rotation in the transform
        camera_lidar_transform.transform.rotation = tf2::toMsg(quaternion);

        tf_broadcaster->sendTransform(camera_lidar_transform);

        geometry_msgs::msg::TransformStamped base_camera_transform;
        base_camera_transform.header.stamp = this->get_clock()->now();
        base_camera_transform.header.frame_id = "base_link";
        base_camera_transform.child_frame_id = "camera_link";

        // Set the translation for base to camera link
        base_camera_transform.transform.translation.x = 0.0;
        base_camera_transform.transform.translation.y = 0.0;
        base_camera_transform.transform.translation.z = 0.0;

        // Convert the extrinsic matrix to a quaternion for the base to camera transform
        tf2::Matrix3x3 rotation_matrix_base(
            extrinsic_matrix(0, 0), extrinsic_matrix(1, 0), extrinsic_matrix(2, 0),
            extrinsic_matrix(0, 1), extrinsic_matrix(1, 1), extrinsic_matrix(2, 1),
            extrinsic_matrix(0, 2), extrinsic_matrix(1, 2), extrinsic_matrix(2, 2));
        rotation_matrix_base.getRotation(quaternion);

        // Set the rotation in the transform
        base_camera_transform.transform.rotation = tf2::toMsg(quaternion);

        tf_broadcaster->sendTransform(base_camera_transform);
    }
};


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto data_publisher = std::make_shared<DataPublisher>();
    rclcpp::spin(data_publisher);
    rclcpp::shutdown();
    return 0;
}
