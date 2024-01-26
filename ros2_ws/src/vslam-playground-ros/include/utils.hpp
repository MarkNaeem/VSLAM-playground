#include <opencv2/opencv.hpp>
#include <opencv2/flann.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <filesystem>

#define DEPTH_HEIGHT_OFFSET 150

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

void projectLidarDataToDepthImage(const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_points,
                                  const Eigen::Matrix4f& extrinsic_matrix,
                                  const Eigen::Matrix3f& intrinsic_matrix,
                                  cv::Mat& depth_image) {
    auto image_height = static_cast<int>(intrinsic_matrix(1, 2))*2;
    auto image_width = static_cast<int>(intrinsic_matrix(0, 2))*2;
    depth_image = cv::Mat::zeros(image_height, image_width, CV_32FC1);

    // Intrinsic Values
    float fx = intrinsic_matrix(0, 0); // Focal length x
    float fy = intrinsic_matrix(1, 1); // Focal length y
    float cx = intrinsic_matrix(0, 2); // Principal point x
    float cy = intrinsic_matrix(1, 2); // Principal point y

    for (const auto& point : *lidar_points)
    {
        // NOTE: intensity needs to be taken out from geometrical calculations
        Eigen::Vector4f lidar_point_homogeneous(point.x, point.y, point.z, 1.0);
        Eigen::Vector4f camera_point_homogeneous = extrinsic_matrix * lidar_point_homogeneous;
        // This is now X Y Z non-homogenous
        Eigen::Vector3f camera_point = camera_point_homogeneous.head<3>() / camera_point_homogeneous(3);

        float px = camera_point(0);
        float py = camera_point(1);
        float pz = camera_point(2);

        if (pz > 0)
        {
            int x = static_cast<int>(std::round( fx * (px/pz) + cx ));
            int y = static_cast<int>(std::round( fy * (py/pz) + cy ));
            if (x >= 0 && x < image_width && y >= DEPTH_HEIGHT_OFFSET && y < image_height) {
                depth_image.at<float>(y, x) = pz;
            }
        }
    }
}


void projectLidarDataToDepthImageFast(const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_points,
                                     const Eigen::Matrix4f& extrinsic_matrix,
                                     const Eigen::Matrix3f& intrinsic_matrix,
                                     cv::Mat& depth_image) {
    auto image_height = static_cast<uint32_t>(intrinsic_matrix(1, 2)) * 2;
    auto image_width = static_cast<uint32_t>(intrinsic_matrix(0, 2)) * 2;
    depth_image = cv::Mat::zeros(image_height, image_width, CV_32FC1);

    // Intrinsic Values
    float fx = intrinsic_matrix(0, 0); // Focal length x
    float fy = intrinsic_matrix(1, 1); // Focal length y
    float cx = intrinsic_matrix(0, 2); // Principal point x
    float cy = intrinsic_matrix(1, 2); // Principal point y

    // Map point cloud (XYZI) to Eigen matrix (XYZ)
    Eigen::Map<const Eigen::Matrix<float, 3, Eigen::Dynamic>> points_mapped(reinterpret_cast<const float*>(lidar_points->points.data()), 3, lidar_points->size());
    // Prepare XYZ1 homogenous matrix
    Eigen::MatrixXf points(4, lidar_points->size());
    // Copy the XYZ part from the XYZI matrix (taken directly from pcl)
    points.topRows<3>() = points_mapped;
    // Add the ones row in XYZ1 
    points.row(3).setOnes();

    // Transform points to camera frame
    Eigen::MatrixXf camera_points_homogeneous = extrinsic_matrix * points;
    Eigen::MatrixXf camera_points = camera_points_homogeneous.topRows<3>();
    camera_points.array().rowwise() /= camera_points_homogeneous.row(3).array();

    // Iterate over valid image points and update depth image
    for (int i = 0; i < camera_points.cols(); ++i) {
        float px = camera_points(0, i);
        float py = camera_points(1, i);
        float pz = camera_points(2, i);      
        if(pz > 0) {
            int x = static_cast<int>(std::round( fx * (px/pz) + cx ));
            int y = static_cast<int>(std::round( fy * (py/pz) + cy ));
            if (x >= 0 && x < image_width && y >= DEPTH_HEIGHT_OFFSET && y < image_height) {
                depth_image.at<float>(y, x) = pz;
            }
        }
    }
}

void projectLidarDataToDepthImageFaster(const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_points,
                                  const Eigen::Matrix4f& extrinsic_matrix,
                                  const Eigen::Matrix3f& intrinsic_matrix,
                                  cv::Mat& depth_image) {
    auto image_height = static_cast<int>(intrinsic_matrix(1, 2))*2;
    auto image_width = static_cast<int>(intrinsic_matrix(0, 2))*2;
    depth_image = cv::Mat::zeros(image_height, image_width, CV_32FC1);

    // Intrinsic Values
    float fx = intrinsic_matrix(0, 0); // Focal length x
    float fy = intrinsic_matrix(1, 1); // Focal length y
    float cx = intrinsic_matrix(0, 2); // Principal point x
    float cy = intrinsic_matrix(1, 2); // Principal point y

    // Extrinsic Values
    float e =  extrinsic_matrix(0,0);
    float r =  extrinsic_matrix(0,1);
    float t =  extrinsic_matrix(0,2);
    float u =  extrinsic_matrix(0,3);
    float d =  extrinsic_matrix(1,0);
    float f =  extrinsic_matrix(1,1);
    float g =  extrinsic_matrix(1,2);
    float h =  extrinsic_matrix(1,3);
    float c =  extrinsic_matrix(2,0);
    float v =  extrinsic_matrix(2,1);
    float b =  extrinsic_matrix(2,2);
    float n =  extrinsic_matrix(2,3);

    for (const auto& point : *lidar_points)
    {
        // Transforming the points from the lidar to the camera frame 
        // This is faster than doing extrnisic * points then dividing by the fourth element to go back from homogenous coords
        float px = point.x*e  + point.y*r  + point.z*t + 1*u;
        float py = point.x*d  + point.y*f  + point.z*g + 1*h;
        float pz = point.x*c  + point.y*v  + point.z*b + 1*n;
        // only take points that have valid z value (depth greater than 0)
        if (pz > 0)
        {
            // Transform the points from the camera coords to the image coords
            // This is faster than doing intrinsic * point then dividing by the third element to normalise and go from 3D to 2D
            int x = static_cast<int>(std::round( fx * (px/pz) + cx ));
            int y = static_cast<int>(std::round( fy * (py/pz) + cy ));
            if (x >= 0 && x < image_width && y >= DEPTH_HEIGHT_OFFSET && y < image_height) {
                depth_image.at<float>(y, x) = pz;
            }
        }
    }
}


void densifyDepthImage(const cv::Mat& depth_image, const double inpaintRadius = 5, const std::string method = "ns") {
    // Create a mask for missing (zero) depth values
    cv::Mat missing_mask = (depth_image == 0);
    missing_mask(cv::Rect(0, 0, depth_image.cols, DEPTH_HEIGHT_OFFSET)) = false;

    // Choose inpainting method
    int flags;
    if (method == "telea") {
        flags = cv::INPAINT_TELEA;
    } else { // Default to "ns" method if anything other than "telea" is specified
        flags = cv::INPAINT_NS;
    }

    // Apply inpainting to fill in the missing depth values
    cv::inpaint(depth_image, missing_mask, depth_image, inpaintRadius, flags);
}


//TODO: Many logic bugs.. doesn't work 
void densifyDepthImageFast(cv::Mat& depth_image) {
    int height = depth_image.rows;
    int width = depth_image.cols;

    // Iterate over the image
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (depth_image.at<float>(y, x) == 0) {
                // Find the nearest non-zero depth value
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


void densifyDepthImageFaster(cv::Mat& depth_image)
{
    // reserve dynamic memory
    const float ratio = 0.04;  
    size_t estimatedNonZeroCount = depth_image.rows * depth_image.cols * ratio; // LiDAR has roughly 4% coverage of the camera FoV 
    size_t estimatedZeroCount    = depth_image.rows * depth_image.cols * (1 - ratio); 
    std::vector<cv::Point2f> nonZeroPoints;
    nonZeroPoints.reserve(estimatedNonZeroCount); // Pre-allocate memory
    std::vector<float> nonZeroValues;
    nonZeroValues.reserve(estimatedNonZeroCount);
    std::vector<cv::Point2f> zeroPoints;
    zeroPoints.reserve(estimatedZeroCount);

    // Step 1: Collect non-zero points and their values
    for (int y = DEPTH_HEIGHT_OFFSET; y < depth_image.rows; ++y) {
        for (int x = 0; x < depth_image.cols; ++x) {
            float value = depth_image.at<float>(y, x);
            if (value != 0) {
                nonZeroPoints.push_back(cv::Point2f(x, y));
                nonZeroValues.push_back(value);
            }
            else{
                zeroPoints.push_back(cv::Point2f(x, y));
            }
        }
    }

    // Convert non-zero points to Mat for FLANN
    cv::Mat nonZeroMat(nonZeroPoints.size(), 2, CV_32F, nonZeroPoints.data());
    // Convert zero points to a single Mat
    cv::Mat zeroPointsMat(zeroPoints.size(), 2, CV_32F, zeroPoints.data());

    // Build FLANN index with non-zero points
    cv::flann::Index flannIndex(nonZeroMat, cv::flann::KDTreeIndexParams(2));

    // Perform a batch knnSearch
    cv::Mat indicesMat(zeroPointsMat.rows, 1, CV_32S); // To store indices of the nearest neighbors
    cv::Mat distsMat(zeroPointsMat.rows, 1, CV_32F); // To store distances to the nearest neighbors

    // Execute batch search
    flannIndex.knnSearch(zeroPointsMat, indicesMat, distsMat, 1, cv::flann::SearchParams(64));

    // Now, indicesMat contains the index of the nearest non-zero point for each zero-valued point
    for (int i = 0; i < zeroPointsMat.rows; ++i) {
        int nearestIndex = indicesMat.at<int>(i, 0);
        const cv::Point2f& queryPoint = zeroPoints[i];
        float nearestValue = nonZeroValues[nearestIndex];
        depth_image.at<float>(queryPoint.y, queryPoint.x) = nearestValue;
    }
}


// Helper function to generate a unique key for each point
inline std::string generateKey(int x, int y) {
    return std::to_string(x) + "_" + std::to_string(y);
}

void densifyDepthImageWithRadius(cv::Mat& depth_image, int radius=3) {
    std::unordered_set<std::string> filled;
    // Iterate over all points
    for (int y = DEPTH_HEIGHT_OFFSET; y < depth_image.rows-radius; ++y) {
        for (int x = radius; x < depth_image.cols-radius; ++x) {
            float value = depth_image.at<float>(y, x);
            if ( value != 0 and filled.find(generateKey(x,y))==filled.end() ) {
                // Fill surrounding initially zero points
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        if(dx==0&&dy==0) continue; //skip the point itself
                        int nx = x + dx;
                        int ny = y + dy;
                        std::string key = generateKey(nx, ny);
                        // if it was filled before, take the average of both fillings 
                        if ( filled.find(key)==filled.end() ) {
                            depth_image.at<float>(ny, nx) = value;
                            filled.insert(key); // Mark as filled to using it for refilling
                        }
                        else {
                            depth_image.at<float>(ny, nx) = 0.5*(depth_image.at<float>(ny, nx)+value);
                        }
                    }
                }
            }
        }
    }
}



void depthImageToPointcloud(const cv::Mat& depth_image, const Eigen::Matrix3f& intrinsic_matrix,  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {

    float fx = intrinsic_matrix(0, 0); // Focal length x
    float fy = intrinsic_matrix(1, 1); // Focal length y
    float cx = intrinsic_matrix(0, 2); // Principal point x
    float cy = intrinsic_matrix(1, 2); // Principal point y

    cloud->width = depth_image.cols;
    cloud->height = depth_image.rows;
    cloud->is_dense = false;
    cloud->points.resize(cloud->width * cloud->height);

    for (int i = DEPTH_HEIGHT_OFFSET; i < depth_image.rows; i++) {
        for (int j = 0; j < depth_image.cols; j++) {
            pcl::PointXYZ& pt = cloud->points[i * depth_image.cols + j];
            
            float depth = depth_image.at<float>(i, j);
            if (std::isnan(depth) || depth <= 0) { // Check for invalid depth
                pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
                continue;
            }

            // Project (i, j, depth) to 3D space
            pt.x = (j - cx) * depth / fx;
            pt.y = (i - cy) * depth / fy;
            pt.z = depth;
        }
    }
}