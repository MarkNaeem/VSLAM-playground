# VSLAM-playground

Building a full Visual SLAM pipeline to experiment with different techniques. The repository also includes a ROS2 interface to load the data from KITTI odometry dataset into ROS2 topics to facilitate visualisation and integration with other ROS2 packages.

## Demos

This video shows visual odometry running on sequence 2 from KITTI odometry dataset. It compares using LightGlue and  SuperGlue as feature matchers. Both are using SuperPoint as a feature extractor and descriptor and depth images created (and densified) from lidar readings.

Odomtery is obtained with PnP. You can see slight drifts the more the sequence goes, but it's worth mentioning that ABSOLUTELY no corrections or loop clousres are used! 

[![Alt text](https://img.youtube.com/vi/QBREBQpEkK4/0.jpg)](https://www.youtube.com/watch?v=QBREBQpEkK4)

This video shows a comparison between different feature extractors and feature matchers running running on the same dataset.

[![Alt text](https://img.youtube.com/vi/ucEH02_uNjE/0.jpg)](https://www.youtube.com/watch?v=ucEH02_uNjE)


## Installation 
- `git clone https://github.com/MarkNaeem/VSLAM-playground.git`
- ` cd VSLAM-payground/`
- `python3 -m pip install -e .` 
- `git submodule init`
- `git submodule update`
- You can directly build the ROS2 package inside the given workspace and source it, or copy the node to your own workspace.
- To build in the given ros2_ws folder, run `colcon build --symlink-install` inside the ros2_ws folder, then `source install/setup.bash`.


**Notes:**
- Make sure to modify `definitions.py` in vslam to point at your KITTI dataset folder.
- Don't forget to go into LightGlue folder and run the install command there `python3 -m pip install -e .`
- Make sure to modify the path to your KITTI dataset folder in `init.launch.py`.


## ROS2 node

Please refer to this [README](./ros2_ws/README.md) for more infromation.