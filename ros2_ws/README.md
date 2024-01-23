## TODO List for ROS 2 Node Improvements

### General Structure and Configuration
- [ ] Modularize the Codebase: Break down the monolithic `DataPublisher` class into smaller, focused classes or modules (e.g., calibration data loader, sensor data processor, ROS publisher).
- [x] Parameterize Configuration: Use ROS 2 parameters for configuring of file paths, data tracks, publishing frequencies, and thread counts.
- [ ] Improve Error Handling: Implement robust error handling throughout the node, particularly where utility functions are called that may throw exceptions.

### Threading and Synchronization
- [ ] Enhance Thread Management: Introduce a graceful shutdown mechanism for worker threads to exit cleanly, using condition variables or similar signaling mechanisms.
- [ ] Ensure Thread Safety: Guard all accesses to shared resources (e.g., `image_map`, `depth_map`, `pointcloud_map`) with mutexes to prevent data races.
- [ ] Optimize Queue Processing: Use condition variables to efficiently manage worker thread wake-up for processing new data entries in queues.

### Data Processing and ROS 2 Practices
- [ ] Validate Calibration Data Loading: Ensure the intrinsic and extrinsic matrices are loaded and applied correctly, including proper handling of the camera matrix in the `createCameraInfoMsg` function.
- [ ] Optimize Data Processing Functions: Review and potentially optimize the lidar to depth image projection and depth image densification functions for performance and accuracy.
- [ ] Publish Static Transforms Once: Modify `publishStaticTransform` to ensure static transforms are published once or at a very low frequency, adhering to best practices for static transforms in ROS 2.

### Code Quality and Maintenance
- [ ] Implement Exception Handling in Threads: Catch exceptions within worker threads to prevent unhandled exceptions from terminating the node unexpectedly.
- [ ] Refactor Hardcoded Values: Replace hardcoded values and paths with configurable parameters or constants defined at the beginning of the file or class.
- [ ] Utilize ROS 2 Features: Explore using more advanced ROS 2 features and best practices, such as lifecycle nodes for managing node states, and services or actions for interactive data processing tasks.

### Testing and Validation
- [ ] Unit Testing: Develop unit tests for critical functionalities, especially the utility functions for data loading and processing, to ensure correctness and robustness.
- [ ] Performance Benchmarking: Benchmark the node's performance, particularly the data processing and publishing rates, under different configurations and workloads to identify bottlenecks.
- [ ] Runtime Configuration Testing: Test dynamic reconfiguration of the node through ROS 2 parameters to ensure the system behaves correctly with varying input parameters.

### Documentation and Comments
- [ ] Update Documentation: Ensure that all functions, especially those within the `utils.hpp` file, are well-documented with comments describing their purpose, parameters, and expected outcomes.
- [ ] Code Comments for Complex Logic: Add detailed comments to complex logic segments, particularly where optimizations are made for performance reasons, to aid future maintenance and understanding.
