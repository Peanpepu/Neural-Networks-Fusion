import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np

class Publisher(Node):
    def __init__(self):
        self.logger = rclpy.logging.get_logger("logger")
        self.logger.info("Inicializa la clase")
        super().__init__("img_pub")
        
        # Create two publishers, one rgb image and another for tir image
        self.rgb_pub = self.create_publisher(Image, "rgb_topic", 10)
        self.tir_pub = self.create_publisher(Image, "term_topic", 10)
        
        # Open video with opencv
        self.video_rgb = cv2.VideoCapture("/home/isa/TFG/right_video_w.mp4")
        self.video_tir = cv2.VideoCapture("/home/isa/TFG/the_video.mp4")
        
        # # Read images
        # img_rgb = cv2.imread("/home/isa/TFG/image_right-1624013356.966232061.jpg")
        # img_tir = cv2.imread("/home/isa/TFG/image_the-1624013357.859271049.jpg")
        
        # # Convert images to a ROS2 message
        # ros_img_rgb = None
        # ros_img_tir = None
        # ros_img_rgb = self.img_conv(img_rgb)
        # ros_img_tir = self.img_conv(img_tir)
        
        # # Publish the images
        # self.rgb_pub.publish(ros_img_rgb)
        # self.tir_pub.publish(ros_img_tir)
        # if (ros_img_rgb != None and ros_img_tir != None):
        #     self.logger.info("Publica imagenes")
        
        
    # Function to convert the cv2 image to ROS2 image    
    def img_conv(self, cv2_img):
        ros_img = Image()
        ros_img.header.stamp = Node.get_clock(self).now().to_msg()
        ros_img.header.frame_id = "ANI717"
        ros_img.height = np.shape(cv2_img)[0]
        ros_img.width = np.shape(cv2_img)[1]
        ros_img.encoding = "bgr8"
        ros_img.is_bigendian = False
        ros_img.step = np.shape(cv2_img)[2] * np.shape(cv2_img)[1]
        ros_img.data = np.array(cv2_img).tobytes()
        return ros_img
        
def main(args=None):
    rclpy.init(args=args)
    node = Publisher()
    node.logger.info("Se inicializa el nodo")
    glob_cont = 0
    
    while rclpy.ok():
        ret_rgb, img_rgb = node.video_rgb.read()
        ret_tir, img_tir = node.video_tir.read()

        if not ret_rgb or not ret_tir:
            break
        
        # Convert images to a ROS2 message
        ros_img_rgb = node.img_conv(img_rgb)
        ros_img_tir = node.img_conv(img_tir)

        # Publish the images
        node.rgb_pub.publish(ros_img_rgb)
        node.tir_pub.publish(ros_img_tir)
        # if (ros_img_rgb != None and ros_img_tir != None):
        #     node.logger.info("Publica imagenes")
            
    # Release the videos
    node.video_rgb.release()
    node.video_tir.release()
    # Cierre del nodo
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()