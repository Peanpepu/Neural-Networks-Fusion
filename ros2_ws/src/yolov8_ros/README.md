# Authory
This project has used as base another project (https://github.com/mgonzs13/yolov8_ros.git) already developed because I had problems to
include all the dependencies correctly. However, all the existing files from this project are unique and different from the original project.

# Using ROS 2 to create a fusion process of neural networks
This project works as a communication system where two different videos (rgb and tir) or images are processed by two neural networks (one with rgb images and another with thermical images) with YOLOV8 and after that, a probabilistic algorithm create a new fusion detection that is sent by ROS2 (the final annotation) so as another computer was able to get the images by one topic and get the final detection with another topic and watch the fusion detection by itself. This project has several python codes in order to test all the process. 
First, img_publisher.py is in charge of sending images from a folder or reading a video to send its frames. 
Second, prueba_ros.py is in charge of reading the images by a ROS2 topic and process them with the neural networks. Then, it creates the fusion detection and send the final annotation by another topic.
Third, visualization.py is in charge of receiving the final annotations and images by ROS topics and show the final result by an opencv image.
The last code called save_img.py was created to read the images directly from a rosbag and save them inside a folder. However, the rosbag I wanted to get was from ROS1 so I couldn't try correctly that code. 

## Installation

```shell
$ cd ~/ros2_ws/src
$ git clone https://github.com/mgonzs13/yolov8_ros.git
$ pip3 install -r yolov8_ros/requirements.txt
$ cd ~/ros2_ws
$ rosdep install --from-paths src --ignore-src -r -y
$ colcon build
```

## Usage

```shell
$ ros2 launch yolov8_bringup prueba_ros.launch.py
```

