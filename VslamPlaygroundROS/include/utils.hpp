#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <filesystem>


int getNumberOfImages(const std::string& directory_path) {
    namespace fs = std::filesystem;
    int file_count = 0;
    for (const auto& entry : fs::directory_iterator(directory_path)) {
        if (entry.is_regular_file()) {
            file_count++;
        }
    }
    return file_count;
}

void loadCalibrationData(const std::string& calib_file, Eigen::Matrix3f &intrinsic_matrix, Eigen::Matrix4f &extrinsic_matrix) {
    std::ifstream file(calib_file);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open calibration file");
    }
    std::string line;
    std::string dump; 
    while (std::getline(file, line)) {
        if (line.find("P0") != std::string::npos) {
            std::istringstream iss(line.substr(3));
            for (int i = 0; i < 12; ++i) {
                if (i==3 || i==7 || i==11)
                {
                    iss >> dump;
                    continue;
                } 
                iss >> intrinsic_matrix(i / 4, i % 4);
            }
        } 
        else if (line.find("Tr") != std::string::npos) {
            std::istringstream iss(line.substr(3));
            extrinsic_matrix.setIdentity();
            for (int i = 0; i < 12; ++i) {
                iss >> extrinsic_matrix(i / 4, i % 4);
            }
        }
    }
}

pcl::PointCloud<pcl::PointXYZI>::Ptr readLidarData(const std::string& path) {
    std::ifstream input_file(path, std::ios::binary);
    if (!input_file.is_open()) {
        throw std::runtime_error("Unable to open lidar file");
    }

    input_file.seekg(0, std::ios::end);
    size_t num_elements = input_file.tellg() / sizeof(float) / 4;
    input_file.seekg(0, std::ios::beg);

    std::vector<float> data(num_elements * 4);
    input_file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    for (size_t i = 0; i < num_elements; ++i) {
        pcl::PointXYZI point;
        point.x = data[i * 4];
        point.y = data[i * 4 + 1];
        point.z = data[i * 4 + 2];
        point.intensity = data[i * 4 + 3];
        cloud->push_back(point);
    }

    return cloud;
}


cv::Mat projectLidarDataToDepthImageFast(const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_points,
                                     const Eigen::Matrix4f& extrinsic_matrix,
                                     const Eigen::Matrix3f& intrinsic_matrix) {
    // Precompute image dimensions
    auto image_height = static_cast<uint32_t>(intrinsic_matrix(1, 2)) * 2;
    auto image_width = static_cast<uint32_t>(intrinsic_matrix(0, 2)) * 2;
    cv::Mat depth_image = cv::Mat::zeros(image_height, image_width, CV_32FC1);

    // Map point cloud to Eigen matrix
    Eigen::Map<const Eigen::Matrix<float, 3, Eigen::Dynamic>> points_mapped(reinterpret_cast<const float*>(lidar_points->points.data()), 3, lidar_points->size());
    Eigen::MatrixXf points(4, lidar_points->size());
    points.topRows<3>() = points_mapped;
    points.row(3).setOnes();

    // Transform points to camera frame
    Eigen::MatrixXf camera_points_homogeneous = extrinsic_matrix * points;
    Eigen::MatrixXf camera_points = camera_points_homogeneous.topRows<3>();
    camera_points.array().rowwise() /= camera_points_homogeneous.row(3).array();

    // Project points onto image plane
    Eigen::MatrixXf image_points_homogeneous = intrinsic_matrix * camera_points;

    // Iterate over valid image points and update depth image
    for (int i = 0; i < image_points_homogeneous.cols(); ++i) {
        if(camera_points(2, i) > 0) {
            int x = static_cast<int>(std::round(image_points_homogeneous(0, i) / image_points_homogeneous(2, i)));
            int y = static_cast<int>(std::round(image_points_homogeneous(1, i) / image_points_homogeneous(2, i)));
            if (x >= 0 && x < image_width && y >= 150 && y < image_height) {
                depth_image.at<float>(y, x) = camera_points(2, i);
            }
        }
    }
    return depth_image;
}


cv::Mat projectLidarDataToDepthImage(const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_points,
                                               const Eigen::Matrix4f& extrinsic_matrix,
                                               const Eigen::Matrix3f& intrinsic_matrix) {
    std::vector<cv::Point3f> image_points;
    auto image_height = static_cast<uint32_t>(intrinsic_matrix(1, 2))*2;
    auto image_width = static_cast<uint32_t>(intrinsic_matrix(0, 2))*2;

    for (const auto& point : *lidar_points) {

        // NOTE: intensity needs to be taken out from geometrical calculations
        Eigen::Vector4f lidar_point_homogeneous(point.x, point.y, point.z, 1.0);
        Eigen::Vector4f camera_point_homogeneous = extrinsic_matrix * lidar_point_homogeneous;
        Eigen::Vector3f camera_point = camera_point_homogeneous.head<3>() / camera_point_homogeneous(3);

        if (camera_point(2) > 0) {
            Eigen::Vector3f image_point_homogeneous = intrinsic_matrix * camera_point;
            cv::Point3f image_point(image_point_homogeneous(0) / image_point_homogeneous(2),
                                    image_point_homogeneous(1) / image_point_homogeneous(2),
                                    camera_point(2));
            if (image_point.x >= 0 && image_point.x < image_width && image_point.y >= 0 && image_point.y < image_height) {
                image_points.push_back(image_point);
            }
        }
    }
    cv::Mat depth_image = cv::Mat::zeros(image_height, image_width, CV_32FC1);
    for (const auto& point : image_points) {
        int x = static_cast<int>(std::round(point.x));
        int y = static_cast<int>(std::round(point.y));
        if (x >= 0 && x < image_width && y >= 0 && y < image_height) {
            depth_image.at<float>(y, x) = point.z;
        }
    }
    return depth_image;
}



void densifyDepthImage(const cv::Mat& depth_image, cv::Mat& densified_depth_image, const std::string& method = "telea") {
    int height = depth_image.rows;
    int width = depth_image.cols;

    // Create a mask for missing (zero) depth values
    cv::Mat missing_mask = (depth_image == 0);
    missing_mask(cv::Rect(0, 0, width, 150)) = false;

    // Choose inpainting method
    double inpaintRadius = 5;  // Inpainting radius
    int flags;
    if (method == "telea") {
        flags = cv::INPAINT_TELEA;
    } else { // Default to "ns" method if anything other than "telea" is specified
        flags = cv::INPAINT_NS;
    }

    // Apply inpainting to fill in the missing depth values
    cv::inpaint(depth_image, missing_mask, densified_depth_image, inpaintRadius, flags);
}


void densifyDepthImageFast(cv::Mat& depth_image) {
    int height = depth_image.rows;
    int width = depth_image.cols;

    // Iterate over the image
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (depth_image.at<float>(y, x) == 0) {
                // Find the nearest non-zero depth value
                int nearest_x = x;
                int nearest_y = y;
                float nearest_depth = 0;
                int search_radius = 1;

                while (nearest_depth == 0 && search_radius < std::max(width, height)) {
                    for (int dy = -search_radius; dy <= search_radius; ++dy) {
                        for (int dx = -search_radius; dx <= search_radius; ++dx) {
                            int nx = x + dx;
                            int ny = y + dy;

                            // Check if within image bounds
                            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                                float depth = depth_image.at<float>(ny, nx);
                                if (depth != 0) {
                                    nearest_x = nx;
                                    nearest_y = ny;
                                    nearest_depth = depth;
                                }
                            }
                        }
                    }
                    ++search_radius;
                }

                // Fill in the missing value
                if (nearest_depth != 0) {
                    depth_image.at<float>(y, x) = nearest_depth;
                }
            }
        }
    }
}

