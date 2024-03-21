from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import TimerAction

import os

def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory('vslam-playground-ros')

    # Define the path to the RViz configuration file
    rviz_config_file = os.path.join(pkg_dir, 'config', 'data_visualiser.rviz')

    # Node to run the data_publisher
    data_node = Node(
        package='vslam-playground-ros',
        executable='data_publisher',
        name='data_publisher',
        parameters=[            
            {'base_directory': "/PATH/TO/YOUR/DATASET"},
            {"sequence_number": "02"},
            {"start_point": 0},            
            {'publish_rate': 10.0},
            {'max_size': 100},
            {'num_image_loader_threads': 1},
            {'num_depth_loader_threads': 3},
            {'depth_densification_method': 'radius'},
            {'densification_radius': 5},
            {'publish_test_pcl': True},
        ],
    )

    # delay to give the odometry node a chance to load 
    # reduce the period if it is too slow for you 
    # or incrfease if the data publisher starts before VO node is loaded 
    delay_action = TimerAction(
        period=5.0,
        actions=[data_node]
    )

    # Node to launch RViz with the specified configuration file
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    # starting the odometry node
    odom_node = Node(
        package='vslam-playground-ros',
        executable='visual_odometry.py',
        name='visual_odometry',
        output='screen'
    )    

    return LaunchDescription([
        odom_node,
        rviz_node,
        delay_action,
    ])
