## Overview

This is a ROS2 wrapper that will be used to demonestrate the VSLAM pipeline in ROS (mostly using Rviz2).

KITTI odometry is the dataset used in this demo. It is efficiently read, handled, and published on ROS2 topics to facilitate running and visualising the rest of the pipeline.   

## TODO List for ROS 2 Node Improvements

- [ ] Implement densification functions in CUDA.
- [ ] Introduce right camera images and camera_info from the dataset.
- [ ] Add depth alignment with the right camera.
- [x] Parameterize Configuration: Use ROS 2 parameters for configuring of file paths, data tracks, publishing frequencies, and thread counts.
- [ ] Optimize Queue Processing: Use condition variables to efficiently manage worker thread wake-up for processing new data entries in queues.
- [x] Add a start point to allow starting from anywhere within the given sequence.
- [ ] Improve Error Handling: Implement robust error handling throughout the node, particularly where utility functions are called that may throw exceptions.
- [ ] Parametrise init.launch.py parameters (so we don't have to modify the file to change them). 
