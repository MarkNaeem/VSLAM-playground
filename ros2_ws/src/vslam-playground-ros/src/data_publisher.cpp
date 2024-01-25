#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <queue>
#include <string>
#include <vector>
#include <algorithm>
#include <mutex>
#include <condition_variable>
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
        // Declare parameters with default values
        this->declare_parameter<std::string>("base_directory");
        this->declare_parameter<std::string>("data_track");
        this->declare_parameter<float>("publish_rate", 10.0); // Publishing rate in Hz
        this->declare_parameter<int>("max_size", 10); // Maximum number of elements in the maps
        this->declare_parameter<int>("num_image_loader_threads", 1);
        this->declare_parameter<int>("num_depth_loader_threads", 2);
        this->declare_parameter<std::string>("depth_densification_method", "none");
        this->declare_parameter<bool>("publish_test_pcl", false);
        this->declare_parameter<int>("densification_radius", 2);

        std::string base_directory;
        std::string data_track;

        // Use parameters to set values
        this->get_parameter("base_directory", base_directory);
        this->get_parameter("data_track", data_track);
        this->get_parameter("publish_rate", publishData_rate);
        this->get_parameter("max_size", MAX_SIZE);
        this->get_parameter("num_image_loader_threads", num_image_loader_threads);
        this->get_parameter("num_depth_loader_threads", num_depth_loader_threads);
        this->get_parameter("depth_densification_method", depth_densification_method);
        this->get_parameter("publish_test_pcl", publish_test_pcl);
        this->get_parameter("densification_radius", densification_radius);
        
        _publishData_millis = static_cast<int>(1000.0 / publishData_rate);

        // Construct paths based on parameters
        image_directory = base_directory + "/sequences/" + data_track + "/image_2";
        lidar_directory = base_directory + "/sequences/" + data_track + "/velodyne";
        calib_file_path = base_directory + "/sequences/" + data_track + "/calib.txt";

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(_publishData_millis),
            std::bind(&DataPublisher::publishData, this));
    
        num_images = getNumberOfImages(image_directory);
        image_pub = this->create_publisher<sensor_msgs::msg::Image>("image_color", 10);
        depth_pub = this->create_publisher<sensor_msgs::msg::Image>("image_depth", 10);
        pointcloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("lidar_points", 10);
        if (publish_test_pcl)
        {
            depth_pcl_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("depth_pcl", 10);
        }
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


        // Populate the ID queue with image indices
        {
            std::lock_guard<std::mutex> depth_lock(depth_id_queue_mutex);
            std::lock_guard<std::mutex> image_lock(image_id_queue_mutex);
            for (int i = 0; i < num_images; ++i)
            {
                depth_id_queue.push(i);
                image_id_queue.push(i);
            }
        }

        // Launch image loader workers
        for (int i = 0; i < num_image_loader_threads; ++i)
        {
            image_loader_threads.emplace_back(&DataPublisher::imageWorker, this);
        }

        // Launch depth loader workers
        for (int i = 0; i < num_depth_loader_threads; ++i)
        {
            depth_loader_threads.emplace_back(&DataPublisher::depthWorker, this);
        }

        publishStaticTransform();
    }


    ~DataPublisher() {
        RCLCPP_INFO(this->get_logger(), "Joining the threads...");
        // Joining all image loader threads
        for (auto& thread : image_loader_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        // Joining all depth loader threads
        for (auto& thread : depth_loader_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }


private:
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr depth_pcl_pub;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_broadcaster;

    rclcpp::TimerBase::SharedPtr timer_;

    Eigen::Matrix3f intrinsic_matrix;
    Eigen::Matrix4f extrinsic_matrix;

    std::unordered_map<int, cv::Mat> image_map;  
    std::unordered_map<int, cv::Mat> depth_map;   
    std::unordered_map<int, sensor_msgs::msg::PointCloud2> pointcloud_map; 

    std::mutex image_map_mutex;
    std::mutex depth_map_mutex;
    std::mutex pointcloud_map_mutex;

    std::queue<int> depth_id_queue;
    std::queue<int> image_id_queue;
    std::mutex depth_id_queue_mutex;
    std::mutex image_id_queue_mutex;

    rclcpp::Clock steady_clock;

    size_t MAX_SIZE; // Maximum number of elements in the maps
    
    int num_image_loader_threads;  // Number of threads for image loading
    int num_depth_loader_threads;  // Number of threads for depth loading

    std::vector<std::thread> image_loader_threads;
    std::vector<std::thread> depth_loader_threads;
    
    bool publish_test_pcl;
    int densification_radius;
    int _publishData_millis;
    float publishData_rate;
    int _data_counter = 0;
    int num_images;
    std::string image_directory;
    std::string lidar_directory;
    std::string calib_file_path;

    std::string depth_densification_method;
    std::unordered_map<std::string,int> densification_methods_map = {{"flann",0},{"radius",1},{"none",2}};


   void imageWorker() {
        RCLCPP_INFO(this->get_logger(), "Image worker thread started with %f Hz.", publishData_rate);
        rclcpp::Rate rate(publishData_rate); 
        while (rclcpp::ok() && !image_id_queue.empty())
        {
            if(image_map.size()<=MAX_SIZE)
            {
                int id;
                {
                    std::unique_lock<std::mutex> lock(image_id_queue_mutex);
                    id = image_id_queue.front();
                    image_id_queue.pop();
                }

                // Load image and process
                std::stringstream ss;
                ss << std::setw(6) << std::setfill('0') << id;
                std::string image_path = image_directory + "/" + ss.str() + ".png";
                cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
                {
                    std::lock_guard<std::mutex> lock(image_map_mutex);
                    image_map[id] = image;
                }
            }
            rate.sleep();
        }
    }


    void depthWorker() {
        RCLCPP_INFO(this->get_logger(), "Depth worker thread started with %f Hz.", 10*publishData_rate);
        rclcpp::Rate rate(publishData_rate); 
        while (rclcpp::ok() && !depth_id_queue.empty())
        {
            if(depth_map.size()<=MAX_SIZE && pointcloud_map.size()<=MAX_SIZE)
            {
                int id;
                {
                    std::unique_lock<std::mutex> lock(depth_id_queue_mutex);
                    id = depth_id_queue.front();
                    depth_id_queue.pop();
                }

                // Load and process depth data
                std::stringstream ss;
                ss << std::setw(6) << std::setfill('0') << id;
                std::string lidar_path = lidar_directory + "/" + ss.str() + ".bin";
                pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_data = readLidarData(lidar_path);
                cv::Mat depth_image;
                projectLidarDataToDepthImageFaster(lidar_data, extrinsic_matrix, intrinsic_matrix, depth_image);
                if (depth_densification_method == "radius") {
                    densifyDepthImageWithRadius(depth_image, densification_radius);
                } else if (depth_densification_method == "flann") {
                    densifyDepthImageFaster(depth_image);
                } else {
                    RCLCPP_INFO_ONCE(this->get_logger(), "No depth image densification method selected or defaulting to none.");
                }
                sensor_msgs::msg::PointCloud2 rosCloud;
                pcl::toROSMsg(*lidar_data, rosCloud);
                rosCloud.header.frame_id = "lidar_link";
                rosCloud.header.stamp = this->get_clock()->now();
                rosCloud.is_dense = false;
                {
                    std::lock_guard<std::mutex> lock_depth(depth_map_mutex);
                    depth_map[id] = depth_image;
                }

                {
                    std::lock_guard<std::mutex> lock_pointcloud(pointcloud_map_mutex);
                    pointcloud_map[id] = rosCloud;
                }
            }
            rate.sleep();
        }
    }


    void publishData() {
        if (_data_counter==0)
        {            
            RCLCPP_INFO_ONCE(this->get_logger(), "Starting data publishing at %f Hz...", 10*publishData_rate);
        }
        
        if (_data_counter >= num_images)
        {
            RCLCPP_INFO_THROTTLE(this->get_logger(), steady_clock, 5000, "Sequence is finished. Please stop the node...");
        }
        else
        {
            cv::Mat color_image;
            cv::Mat depth_image;
            sensor_msgs::msg::PointCloud2 pointcloud;
            {
                std::lock_guard<std::mutex> image_lock(image_map_mutex);
                std::lock_guard<std::mutex> depth_lock(depth_map_mutex);
                std::lock_guard<std::mutex> pointcloud_lock(pointcloud_map_mutex);
                // TODO: Modify this with a while to keep trying to get the image, depth, and pointcloud for the entire length of the time window consistently 
                if (image_map.find(_data_counter) != image_map.end() &&
                    depth_map.find(_data_counter) != depth_map.end() &&
                    pointcloud_map.find(_data_counter) != pointcloud_map.end())
                {
                    RCLCPP_INFO(this->get_logger(), "Publishing data for image %d of %d", _data_counter + 1, num_images);

                    auto imgs_header = std_msgs::msg::Header();
                    imgs_header.frame_id = "camera_frame";
                    imgs_header.stamp = this->get_clock()->now();

                    color_image = image_map[_data_counter];
                    auto ros_image = cv_bridge::CvImage(imgs_header, "bgr8", color_image).toImageMsg();
                    depth_image = depth_map[_data_counter];
                    auto ros_depth = cv_bridge::CvImage(imgs_header, "32FC1", depth_image).toImageMsg();
                    pointcloud = pointcloud_map[_data_counter];
                    image_pub->publish(*ros_image);
                    depth_pub->publish(*ros_depth);
                    pointcloud_pub->publish(pointcloud);
                    
                    //cleaning the memory
                    image_map.erase(_data_counter);
                    depth_map.erase(_data_counter);
                    pointcloud_map.erase(_data_counter);

                    // prepare for next iteration
                    _data_counter++;
                }
                else
                {
                    RCLCPP_INFO_THROTTLE(this->get_logger(), steady_clock, 1000, "Waiting for data...");
                }
            }
            if (publish_test_pcl)
            {
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
                depthImageToPointcloud(depth_image, intrinsic_matrix, cloud);                
                // Convert to ROS message
                sensor_msgs::msg::PointCloud2 output;
                pcl::toROSMsg(*cloud, output);
                output.header.frame_id = "camera_link"; 
                output.header.stamp = this->get_clock()->now();
                // Publish the point cloud
                depth_pcl_pub->publish(output);
            }            
            sensor_msgs::msg::CameraInfo camera_info_msg = createCameraInfoMsg();
            camera_info_pub->publish(camera_info_msg);
        }
    }

    sensor_msgs::msg::CameraInfo createCameraInfoMsg() {
        sensor_msgs::msg::CameraInfo camera_info_msg;
        camera_info_msg.header.frame_id = "camera_link";
        camera_info_msg.header.stamp = this->get_clock()->now();
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
