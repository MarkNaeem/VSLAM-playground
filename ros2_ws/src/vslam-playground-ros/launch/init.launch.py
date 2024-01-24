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
            {'publish_rate_millis': 100},
            {'max_size': 50},
            {'num_image_loader_threads': 1},
            {'num_depth_loader_threads': 2},
            {'densify_depth': True},
            {'publish_test_pcl': True},
            {'densification_radius': 2},
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
