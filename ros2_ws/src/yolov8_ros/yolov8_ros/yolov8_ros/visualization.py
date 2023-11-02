
from ultralytics import YOLO
#import yolo
import cv2
import numpy as np
import threading
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from yolov8_msgs.msg import Detection
from yolov8_msgs.msg import DetectionArray
from std_msgs.msg import String
from cv_bridge import CvBridge



class Visualization(Node):
    def __init__(self):
        self.logger = rclpy.logging.get_logger("logger")
        self.logger.info("Inicializa la clase")
        super().__init__("visualization")
        self.cv_bridge = CvBridge()
        self.img_rgb = None # Atributo para almacenar la imagen rgb convertida a opencv
        self.img_term = None # Atributo para almacenar la imagen rgb convertida a opencv
        self.res_tir = None # Atributo para almacenar la imagen térmica resultado convertida a opencv
        self.res_rgb = None # Atributo para almacenar la imagen rgb resultado convertida a opencv
        self.data = None # Atributo para almacenar las anotaciones
        self.subscription_rgb = self.create_subscription(Image, "rgb_topic", self.image_rgb_callback, 10)
        self.subscription_rgb = self.create_subscription(Image, "term_topic", self.image_tir_callback, 10)
        self.subs_res_rgb = self.create_subscription(Image, "rgb_res_img", self.rgb_callback, 10)
        self.subs_res_tir = self.create_subscription(Image, "tir_res_img", self.tir_callback, 10)
        self.subscription_bboxes = self.create_subscription(DetectionArray, "annotations_topic", self.annotations_callback, 10)
        
        

    def image_rgb_callback(self, msg):
        # Procesa la imagen rgb original recibida
        self.img_rgb = self.cv_bridge.imgmsg_to_cv2(msg)
        # self.logger.info("Convierte rgb img")
    
    def image_tir_callback(self, msg):
        # Procesa la imagen térmica original recibida
        self.img_term = self.cv_bridge.imgmsg_to_cv2(msg)
        # self.logger.info("Convierte rgb img")
        
    def rgb_callback(self, msg):
        # Procesa la imagen resultado recibida
        self.res_rgb = self.cv_bridge.imgmsg_to_cv2(msg)
        
    def tir_callback(self, msg):
        # Procesa la imagen resultado recibida
        self.res_tir = self.cv_bridge.imgmsg_to_cv2(msg)
        # self.logger.info("Convierte term img")

    def annotations_callback(self, annotations):
        # Recibe las anotaciones de la fusión
        self.data = annotations

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

    ############################################################################################

    # Función donde se leen las anotaciones de una red y se devuelven los objetos detectados
    def read_annotations(self, img):
        # Obtener las dimensiones de la imagen
        height, width, _ = img.shape
        
        # Lista para almacenar la ubicación y el área de cada objeto detectado
        detected_objects = []

        for annotation in self.data.detections:
            # self.logger.info("Cantidad de objetos detectados: " + str(len(self.data.detections)))
            
            # Obtener la información de cada anotación
            class_id = annotation.class_id
            x_center = annotation.bbox.center.x
            y_center = annotation.bbox.center.y
            bbox_width = annotation.bbox.size.w
            bbox_height = annotation.bbox.size.h
            conf = round(annotation.conf, 2) # Se redondea la confianza a 2 decimales

            # Calcular las coordenadas del bounding box
            x = int((x_center - bbox_width/2) * width) # Se multiplica porque las coordenadas del bbox son relativas al ancho y alto del mismo
            y = int((y_center - bbox_height/2) * height)
            w = int(bbox_width * width)
            h = int(bbox_height * height)
            
            # Almacenar al ubicación y el área del objeto detectado en la lista
            detected_objects.append((x, y, w, h, int(class_id), conf))
        return detected_objects

    ##########################################################################################

    def draw_bounding_boxes(self, img, detected_objects, cont2):
        
        # Declaramos las clases con sus índices correspondientes
        classes = {
            0: 'emergency vehicle',
            1: 'non-emergency vehicle',
            2: 'first responder',
            3: 'non-first responder'
        }
        
        # Definir los colores para cada clase
        colors = [(73, 164, 73), (183, 56, 56), (0, 0, 255), (196, 196, 10)]
        cont = 0
        
        for obj in detected_objects:
            cont = cont+1
            # self.logger.info("Bbox del objeto " + str(cont))
            x, y, w, h, class_id, conf = map(float, obj)            
            
            # Se obtiene el color correspondiente a cada clase
            color = colors[int(class_id)]
            
            # Dibujar el bounding box en la imagen
            cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)

            # Obtener el texto y el color para el texto
            text = classes[int(class_id)] + " " + str(conf)
            text_color = (255, 255, 255)  # Color blanco

            # Calcular el tamaño del texto
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

            # Calcular las coordenadas para el rectángulo del texto
            rect_x = int(x)
            rect_y = int(y) - text_size[1] - 10
            rect_width = text_size[0] + 10
            rect_height = text_size[1] + 10

            # Dibujar el rectángulo del texto
            cv2.rectangle(img, (rect_x, rect_y), (rect_x+rect_width, rect_y+rect_height), color, -1)

            # Dibujar el texto con el color correspondiente
            cv2.putText(img, text, (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
            
            

        # Guardamos la imagen resultante con los boundig boxes
        output_path = "/home/isa/TFG/images_res/image" + str(cont2) + ".jpg"
        cv2.imwrite(output_path, img)
        
        # Devolvemos la imagen resultante con los boundig boxes
        return img

#######################################################################################################












def main(args=None):
    rclpy.init(args=args)
    node = Visualization()
    node.logger.info("Se inicializa el nodo")
    # Crea la ventana
    cv2.namedWindow("Red fusionada", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Red RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Red térmica", cv2.WINDOW_NORMAL)
    #node.logger.info("Llega aquí 0")
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
        if (node.res_rgb is not None and node.res_tir is not None): #and (node.img_rgb != last_rgb and node.img_term != last_term)):

            #node.logger.info("Ambas imágenes enviadas")
            
            image_rgb = node.crop_image(node.img_rgb)
            # image_term = node.img_term
            last_rgb = image_rgb
            # last_term = image_term
            if node.data is not None:
                # Leemos las predicciones hechas por las redes
                objects = node.read_annotations(image_rgb) # , node.data) 
                # Solo recibe la imagen como entrada porque las anotaciones se le pasan directamente como atributo de la clase
                # objects_term = node.read_annotations(image_term, "/home/isa/ros2_pedro/TFG/TIR_network/labels/image0.txt")
                #node.logger.info("Leo anotaciones")
                # Dibujamos los bboxes en la imagen
                img = node.draw_bounding_boxes(image_rgb, objects, cont2)
                cont2 = cont2+1
                # Mostramos la imagen
                cv2.imshow("Red fusionada", img)
                cv2.imshow("Red RGB", node.res_rgb)
                cv2.imshow("Red térmica", node.res_tir)
                # Espera un breve periodo de tiempo y verifica si se presiona la tecla 'q' para salir
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            

        else:
            time.sleep(1)




    # Cierre del nodo
    # cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()




        

