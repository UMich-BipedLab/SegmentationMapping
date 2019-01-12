# segmentation_projection
Project lidar points into camera frame, and then label them according to the pixel class predicted by the neural nets.

# dependencies
`ros_numpy`, `tensorflow-gpu-1.8.0`, `python-opencv`, `cuda-9.0`

# launch
`roslaunch segmentation_projection realsense.launch`

# parameters
* `neural_net_graph`: The path of the neural network graph.pb file
* `is_output_distribution`: whether we need the distribution of all classes, or just the final label (the class with the max probability)
* `neural_net_input_width`: the width of the neural network input
* `neural_net_input_height`: the height of the neural network input
* `lidar`: the topic of lidar Pointcloud2
* `camera_num`: the number of cameras
* `image_0`: the image topic of 0-th camera. Use `image_[0-9]` to indexing camera topics
* `cam_intrinsic_0`: the `npy` file containing the intrinsic transformation of 0-th camera. Use `image_[0-9]` to indexing camera topics
*  `cam2lidar_file_0`: the `npy` file containing camera to lidar transformation of 0-th camera. Use `image_[0-9]` to indexing camera topics

