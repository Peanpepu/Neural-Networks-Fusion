from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    prueba_ros = Node(
		package="yolov8_ros",
		executable="prueba_ros",
		output="screen")
    
    img_publisher = Node(
		package="yolov8_ros",
		executable="img_publisher")
    
    visualization = Node(
		package="yolov8_ros",
		executable="visualization")
    
    ld = LaunchDescription()
    ld.add_action(img_publisher)
    ld.add_action(prueba_ros)
    ld.add_action(visualization)
    
    return ld
