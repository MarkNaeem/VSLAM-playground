from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory('vslam-playground-ros')

    # Define the path to the RViz configuration file
    rviz_config_file = os.path.join(pkg_dir, 'config', 'data_visualiser.rviz')

    # Node to run the data_publisher
    vslam_node = Node(
        package='vslam-playground-ros',
        executable='data_publisher',
        name='data_publisher',
        parameters=[            
            {'base_directory': "/media/mark/New Volume/kitti-dataset/"},
            {"data_track": "02"},
            {'publish_rate': 10.0},
            {'max_size': 100},
            {'num_image_loader_threads': 2},
            {'num_depth_loader_threads': 4},
            {'depth_densification_method': 'radius'},
            {'densification_radius': 3},
            {'publish_test_pcl': False},
        ],
    )

    # Node to launch RViz with the specified configuration file
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    return LaunchDescription([
        vslam_node,
        rviz_node,
    ])
