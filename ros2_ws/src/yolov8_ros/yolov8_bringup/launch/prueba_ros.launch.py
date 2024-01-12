from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    fusion_ros = Node(
		package="yolov8_ros",
		executable="fusion_ros",
		output="screen")
    
    img_publisher = Node(
		package="yolov8_ros",
		executable="img_publisher")
    
    visualization = Node(
		package="yolov8_ros",
		executable="visualization")
    
    ld = LaunchDescription()
    ld.add_action(img_publisher)
    ld.add_action(fusion_ros)
    ld.add_action(visualization)
    
    return ld
