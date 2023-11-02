
import cv2
import numpy as np
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge



class Visualization(Node):
    def __init__(self):
        self.logger = rclpy.logging.get_logger("logger")
        self.logger.info("Inicializa la clase")
        super().__init__("visualization")
        self.cv_bridge = CvBridge()
        self.img_rgb = None # Atributo para almacenar la imagen rgb convertida a opencv
        self.img_term = None # Atributo para almacenar la imagen rgb convertida a opencv
        self.subscription_rgb = self.create_subscription(Image, "/zed2/zed_node/right_raw/image_raw_color", self.image_rgb_callback, 10)
        self.subscription_rgb = self.create_subscription(Image, "/axis_the/camera/image_raw/compressed", self.image_tir_callback, 10)
        
        

    def image_rgb_callback(self, msg):
        # Procesa la imagen rgb original recibida
        self.img_rgb = self.cv_bridge.imgmsg_to_cv2(msg)
        # self.logger.info("Convierte rgb img")
    
    def image_tir_callback(self, msg):
        # Procesa la imagen térmica original recibida
        self.img_term = self.cv_bridge.imgmsg_to_cv2(msg)
        # self.logger.info("Convierte rgb img")
        
    
    ############################################################################################

    # Función donde se recorta la imagen rgb para adaptarla a la térmica
    def crop_image(self, image):

        # Se establecen las dos esquinas del recorte
        points = [(448, 469), (840, 169)]

        # Obtener las coordenadas de los dos puntos seleccionados
        (x1, y1), (x2, y2) = points

        # Recortar la imagen utilizando los dos puntos seleccionados
        cropped_image = image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

        # Redimensionar el recorte al tamaño de la imagen térmica
        term_size = (704, 576)
        resized_image = cv2.resize(cropped_image, term_size)

        return resized_image

#######################################################################################################

def main(args=None):
    rclpy.init(args=args)
    node = Visualization()
    node.logger.info("Se inicializa el nodo")
    cont2 = 0

    while rclpy.ok():
        # Ejecuta las tareas dentro del nodo
        #node.logger.info("Llega aquí 1")
        # No hace falta el spin_once porque ya está subscrito a los topic y se bloquea esperando a terminar de escuchar
        rclpy.spin_once(node)#, timeout_sec = 0.5)
        if node.img_rgb is None:
            node.logger.info("Img_rgb vacía")
        if node.img_term is None:
            node.logger.info("Img_term vacía")
        
        # Si no se han transmitido imágenes nuevas (valor inicial = None) no ejecuta esto
        if (node.img_rgb is not None and node.img_term is not None): #and (node.img_rgb != last_rgb and node.img_term != last_term)):

            #node.logger.info("Ambas imágenes enviadas")
            image_rgb = node.crop_image(node.img_rgb)
            cv2.imwrite("/isa/data/rosbag_pedro/2023/images/RGB/img" + str(cont2) + ".jpg", image_rgb)
            cv2.imwrite("/isa/data/rosbag_pedro/2023/images/TIR/img" + str(cont2) + ".jpg", node.img_term)
            cont2 = cont2+1  
            
        else:
            time.sleep(1)




    # Cierre del nodo
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()




        

