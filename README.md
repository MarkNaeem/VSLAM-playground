# VSLAM-playground
Building a full Visual SLAM pipeline to experiment with different techniques

## Installation 
- `git clone https://github.com/MarkNaeem/VSLAM-playground.git`
- ` cd VSLAM-payground/`
- `python3 -m pip install -e .` 
- `git submodule init`
- `git submodule update`
- You can directly build the ROS2 package inside the given workspace and source it, or copy the node to your own workspace.
- To build in the given ros2_ws folder, run `colcon build --symlink-install` inside the ros2_ws folder, then `source install/setup.bash`.


**Notes:**
- Make sure to modify definitions.py to point at your KITTI dataset. (**UPDATE:** Now the C++ node has this path as a parameter.)
- Don't forget to go into LightGlue folder and run the install command there `python3 -m pip install -e .`


## ROS2 node

Please refer to this [README](./ros2_ws/README.md) for more infromation.