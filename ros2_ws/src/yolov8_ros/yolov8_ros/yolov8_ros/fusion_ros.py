import shutil
import os
from ultralytics import YOLO
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



class ImgPubSub(Node):
    def __init__(self):
        self.logger = rclpy.logging.get_logger("logger")
        self.logger.info("Inicializa la clase")
        super().__init__("img_pub_sub")
        self.cv_bridge = CvBridge()
        self.img_rgb = None # Atributo para almacenar la imagen rgb convertida a opencv
        self.img_term = None # Atributo para almacenar la imagen térmica convertida a opencv
        self.subscription_rgb = self.create_subscription(Image, "rgb_topic", self.image_rgb_callback, 10)
        self.subscription_tir = self.create_subscription(Image, "term_topic", self.image_term_callback, 10)
        self.publisher = self.create_publisher(DetectionArray, "annotations_topic", 10)
        self.publisher_rgb_img = self.create_publisher(Image, "rgb_res_img", 10)
        self.publisher_tir_img = self.create_publisher(Image, "tir_res_img", 10)
        
        

    def image_rgb_callback(self, msg):
        # Procesa la imagen recibida y obtén la lista de listas
        self.img_rgb = self.cv_bridge.imgmsg_to_cv2(msg)
        #self.logger.info("Convierte rgb img")
        
    def res_rgb_callback(self, msg):
        # Procesa la imagen recibida y obtén la lista de listas
        self.publisher_imgs = self.cv_bridge.imgmsg_to_cv2(msg)
    
    def image_term_callback(self, msg):
        # Procesa la imagen recibida y obtén la lista de listas
        self.img_term = self.cv_bridge.imgmsg_to_cv2(msg)
        #self.logger.info("Convierte term img")

    def res_tir_callback(self, msg):
        # Procesa la imagen recibida y obtén la lista de listas
        self.img_term = self.cv_bridge.imgmsg_to_cv2(msg)
        
    def publish_callback(self, annotations, rgb_img, tir_img):
        self.publisher.publish(annotations)
        self.publisher_rgb_img.publish(rgb_img)
        self.publisher_tir_img.publish(tir_img)

    ############################################################################################

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
    
    ################################################################################################

    # Función donde se leen las anotaciones de una red y se devuelven los objetos detectados
    def read_annotations(self, img, annotations_file):
        # Obtener las dimensiones de la imagen
        height, width, _ = img.shape

        # Leer las anotaciones desde el archivo de anotaciones YOLO
        with open(annotations_file, 'r') as f:
            annotations = f.read().splitlines()
        
        # Lista para almacenar la ubicación y el área de cada objeto detectado
        detected_objects = []

        # Si se ha detectado algún objeto en la imagen:
        try:
            # Leer las anotaciones desde el archivo de anotaciones YOLO
            with open(annotations_file, 'r') as f:
                annotations = f.read().splitlines()
                
            for annotation in annotations:

                # Obtener la información de cada anotación
                class_id, x_center, y_center, bbox_width, bbox_height, conf = map(float, annotation.split())
                conf = round(conf, 2) # Se redondea la confianza a 2 decimales

                # Calcular las coordenadas del bounding box
                x = int((x_center - bbox_width/2) * width) # Se multiplica porque las coordenadas del bbox son relativas al ancho y alto del mismo
                y = int((y_center - bbox_height/2) * height)
                w = int(bbox_width * width)
                h = int(bbox_height * height)
                
                # Almacenar al ubicación y el área del objeto detectado en la lista
                detected_objects.append(((x, y, w, h), class_id, conf))
            
        except:
            detected_objects.append(((0, 0, 0, 0), 5, 0))
            
        return detected_objects

    ############################################################################################

    # Función para checkear la interseccion de dos bounding boxes de ambas redes
    def check_intersection(self, rgb_object, index, term_objects, op=60):
        # Crear una lista vacía para apuntar las coincidencias y sus porcentajes de superposición
        coincidencias = []
        ((x1, y1, w1, h1), _, _) = rgb_object
        # Calcular el área del objeto actual
        area1 = w1 * h1
        # Contador para comprobar si más de un objeto cumple la intersección mínima
        cont = 0
        for j, ((x2, y2, w2, h2), _, _) in enumerate(term_objects):
            # Calcular el área del objeto previo
            area2 = w2 * h2
        
            # Calcular la intersección entre los objetos 
            intersection_x = max(x1, x2)
            intersection_y = max(y1, y2)
            intersection_w = min(x1+w1, x2+w2) - intersection_x
            intersection_h = min(y1+h1, y2+h2) - intersection_y
        
            # Comprobamos que el IoU sea mayor que 0
            if (intersection_w > 0 and intersection_h > 0):
                # Calcular el área de la intersección
                intersection_area = intersection_w * intersection_h
            
                # Calcular el porcentaje de superposición
                union_area = float(area1 + area2 - intersection_w*intersection_h)
                overlap_percentage = round(intersection_area / union_area * 100, 2)
            
                if overlap_percentage >= op:
                    cont = cont + 1
                    # Mostrar el porcentaje de superposición y guardarlo
                    # # print(f"Objeto {index+1} y Objeto {j+1}: {overlap_percentage}% de superposición")
                    # Si varios objetos tienen un IoU > 60% se escoge el mayor
                    if cont > 1:
                        if overlap_percentage > coincidencias[0][2]:
                            del coincidencias[0]
                            objects = (index, j, overlap_percentage)
                            coincidencias.append(objects)
                    else:
                        objects = (index, j, overlap_percentage)
                        coincidencias.append(objects)
                    

        return coincidencias

    ##############################################################################################

    # Función para comprobar si un mismo objeto se ha detectado con varias bboxes
    def check_objects(self, objects):
        # Se crea una lista vacía para las posiciones de los objetos duplicados
        list_dup = []
        # Se recorren los objetos finales buscando aquellos de la misma clase con un IoU superior al 30%
        for i, (class1, x1, y1, w1, h1, conf1) in enumerate(objects):
            # Si este objeto no está en la lista de duplicados se ejecuta el siguiente código
            if i not in list_dup and i < len(objects)-1:
                # Calcular el área del objeto actual
                area1 = w1 * h1
                for j, (class2, x2, y2, w2, h2, conf2) in enumerate(objects[i+1:]):
                    # Si este objeto no está en la lista de duplicados se ejecuta el siguiente código
                    if j not in list_dup:
                        # Calcular el área del objeto previo
                        area2 = w2 * h2
                    
                        # Calcular la intersección entre los objetos 
                        intersection_x = max(x1, x2)
                        intersection_y = max(y1, y2)
                        intersection_w = min(x1+w1, x2+w2) - intersection_x
                        intersection_h = min(y1+h1, y2+h2) - intersection_y
                    
                        # Comprobamos que el IoU sea mayor que 0
                        if (intersection_w > 0 and intersection_h > 0):
                            # Calcular el área de la intersección
                            intersection_area = intersection_w * intersection_h
                        
                            # Calcular el porcentaje de superposición
                            union_area = float(area1 + area2 - intersection_w*intersection_h)
                            overlap_percentage = round(intersection_area / union_area * 100, 2)
                        
                            if overlap_percentage >= 30:
                                if (class1 < 2 and class2 < 2) or (class1 > 1 and class2 > 1):
                                    # Mostrar el porcentaje de superposición y guardarlo
                                    # print(f"Objeto final {i} y Objeto final {j+1}: {overlap_percentage}% de superposición")
                                    # Si varios objetos tienen un IoU > 30% se escoge el de mayor confianza
                                    if conf1 >= conf2:
                                        list_dup.append(j+1)
                                    else:
                                        list_dup.append(i)

        return list_dup

    ##########################################################################################

    # Función para escribir el objeto detectado en una anotación
    def write_annotation(self, objects, output_path):

        # Escribir cada objeto de la lista en una línea del archivo
        with open(output_path, "w") as file:
            for obj in objects:
                # " ".join() une todo lo que haya dentro del paréntesis con un espacio
                linea = " ".join(str(element) for element in obj)  # Convertir cada elemento a cadena y unirlos con espacio
                file.write(linea + "\n")  # Escribir la línea en el archivo y agregar un salto de línea

        print("Archivo creado exitosamente.")

    ##########################################################################################















def main(args=None):
    rclpy.init(args=args)
    node = ImgPubSub()
    node.logger.info("Se inicializa el nodo")
    
    # Borramos la carpeta donde se van a guardar los datos de la detección para que no haya errores
    try:
        shutil.rmtree("/home/isa/ros2_pedro/TFG")
    except:
        pass

    # Path para la imagen rgb: /home/dgx/datasets/isa/rescate/raw/bag/extracted_dataset_2020_2021/2021/2021-06-18-12-49-08/image_right/image_right-1624013356.966232061.jpg
    # Path para la imagen térmica: /home/dgx/datasets/isa/rescate/raw/bag/extracted_dataset_2020_2021/2021/2021-06-18-12-49-08/image_the/image_the-1624013357.859271049.jpg
    # Cargar los modelos entrenados
    model_rgb = YOLO("/home/isa/TFG/results/train_rgb/weights/best.pt") # Ruta del archivo .pt obtenido tras el entrenamiento
    model_term = YOLO("/home/isa/TFG/results/train_term/weights/best.pt")
    
    last_rgb = None
    last_term = None

    #node.logger.info("Llega aquí 0")

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

            # node.logger.info("Llega aquí 1")
            
            image_rgb = node.crop_image(node.img_rgb)
            image_term = node.img_term
            last_rgb = image_rgb
            last_term = image_term

            # Realizar las detecciones en las imagenes
            results_rgb = model_rgb.predict(project="TFG", name="RGB_network", stream=False, source=image_rgb, save=True, save_txt=True, save_conf=True)
            results_term = model_term.predict(project="TFG", name="TIR_network", stream=False, source=image_term, save=True, save_txt=True, save_conf=True)
            
            # save=True guarda la imagen resultado en ultralytics/runs/detect/predict2
            # save_txt=True guarda el resultado como un archivo txt
            # save_conf=True guarda el resultado con los valores de confianza de cada predicción
            # stream=True ahorra memoria para recibir muchas imágenes y dar resultados más rápidos pero no muestra las anotaciones

            ### A continuación, se crea el algoritmo de la fusión
            
            # Leemos las predicciones hechas por las redes
            objects_rgb = node.read_annotations(image_rgb, "/home/isa/ros2_pedro/TFG/RGB_network/labels/image0.txt")
            objects_term = node.read_annotations(image_term, "/home/isa/ros2_pedro/TFG/TIR_network/labels/image0.txt")
            print("Leo anotaciones")
            
            # Declaramos las clases con sus índices correspondientes
            classes = {
                0: 'emergency vehicle',
                1: 'non-emergency vehicle',
                2: 'first responder',
                3: 'non-first responder',
                5: '' # Vacío, ninguna detección encontrada
            }
            objects = []
            # Inicializamos las coordenadas resultado en todo 0
            (x, y, w, h, prob) = (0, 0, 0, 0, 0)
            # Se obtiene el ancho y largo de las imagenes (tienen las mismas dimensiones) para posteriores conversiones
            height, width, _ = image_rgb.shape
            # Lista con los objetos térmicos detectados en ambas redes
            shared_term_obj = []
            # Lista con las detecciones finales
            msg_array = DetectionArray()
            
            # El objeto viene representado así: ((x, y, w, h), class_id)

            # Identificamos los objetos detectados por ambas redes y separamos cada tipo de detección
            for i, object_rgb in enumerate(objects_rgb):
                match_object = node.check_intersection(object_rgb, i, objects_term)
                print("Checkeamos intersección")
                
                # Extraemos los datos del objeto rgb
                (x1, y1, w1, h1), class_rgb, conf_rgb = object_rgb
                # Reconvertimos las coordenadas al centro del bbox rgb
                xc1 = x1 + w1/2
                yc1 = y1 + h1/2
                # Si ambas redes han detectado un mismo bbox (objeto)
                if match_object != []:
                    j = match_object[0][1] # Índice del objeto térmico que se ha detectado que es igual
                    shared_term_obj.append(j)
                    # Extraemos los datos del objeto térmico
                    (x2, y2, w2, h2), class_term, conf_term = objects_term[j]
                    # Reconvertimos las coordenadas al centro del bbox térmico
                    xc2 = x2 + w2/2
                    yc2 = y2 + h2/2
                    # Calculamos la confianza máxima y la mínima obtenida entre las dos redes
                    p = max(conf_rgb, conf_term)
                    q = min(conf_rgb, conf_term)
                    
                    # Si ninguna red a detectado ningún objeto (todos los valores a 0 excepto la clase que es la 5)
                    if class_rgb == 5 and class_term == 5:
                        (x, y, w, h, ob_class, prob) = (0, 0, 0, 0, 5, 0)
                        
                    # Ahora separamos si coinciden en la clase raíz
                    elif (class_rgb < 2 and class_term < 2) or (class_rgb > 1 and class_term > 1):
                        # Ahora separamos si coinciden en la clase específica
                        if class_rgb == class_term:
                            ob_class = class_rgb
                            # Calculamos la confianza
                            prob = round(p + (1-p) * q, 2)

                            # Escogemos el bbox con mayor confianza inicial
                            if conf_rgb >= conf_term:
                                (x, y, w, h) = (xc1/width, yc1/height, w1/width, h1/height)
                            else:
                                (x, y, w, h) = (xc2/width, yc2/height, w2/width, h2/height)
                        
                        # Separamos si coinciden en clase raíz pero no en específica
                        else:
                        # Calculamos la confianza
                            prob = round(((p + (1-p) * q) + (p - p*q)) / 2, 2)
                            # Escogemos el bbox y clase con mayor confianza inicial
                            if conf_rgb >= conf_term:
                                (x, y, w, h, ob_class) = (xc1/width, yc1/height, w1/width, h1/height, class_rgb)
                            else:
                                (x, y, w, h, ob_class) = (xc2/width, yc2/height, w2/width, h2/height, class_term)
                    
                    # Caso donde no coinciden en clase raíz. Este caso no debería ocurrir pero se incluye por seguridad
                    else:
                        # Calculamos la confianza
                        prob = p - p*q
                        # Escogemos el bbox y la probabilidad final del objeto con mayor confianza
                        if conf_rgb > conf_term:
                            (x, y, w, h) = (xc1/width, yc1/height, w1/width, h1/height)
                            ob_class = class_rgb
                        else:
                            (x, y, w, h) = (xc2/width, yc2/height, w2/width, h2/height)
                            ob_class = class_term
                        # print("Probabilidad obtenida en mismo objeto pero distinta clase es " + str(prob))
                        
                        
                # Detecta objeto red rgb y no la red térmica
                elif class_rgb != 5:        
                    # Se mantiene la confianza de la red rgb
                    prob = conf_rgb
                    # Guardamos las coordenadas y dimensiones del bbox
                    (x, y, w, h) = (xc1/width, yc1/height, w1/width, h1/height)
                    ob_class = class_rgb

                objects.append([ob_class, x, y, w, h, prob])
                
            # Ahora recogemos el caso en que el objeto ha sido detectado por la red térmica pero no por la rgb
            for i, object_term in enumerate(objects_term):
                # Se comprueba si ninguna red ha detectado ningún objeto
                if int(object_term[1]) == 5:
                    # Se guarda el objeto en una variable
                    objects.append([5, 0, 0, 0, 0, 0])
                    
                # Si ha detectado objetos y no son iguales a los rgb
                elif i not in shared_term_obj:
                    # Extraemos los datos del objeto térmico
                    (x2, y2, w2, h2), ob_class, conf_term = object_term
                    # Reconvertimos las coordenadas al centro del bbox térmico
                    xc2 = x2 + w2/2
                    yc2 = y2 + h2/2
                    # Se mantiene la confianza de la red térmica
                    prob = conf_term
                    (x, y, w, h) = (xc2/width, yc2/height, w2/width, h2/height)
                    
                    # Se guarda el objeto en una variable
                    objects.append([ob_class, x, y, w, h, prob])

            # Ahora se checkea que no haya varios objetos superpuestos para un solo objeto real
            list_dup = node.check_objects(objects)
            list_dup.sort()
            # Se crea un contador para controlar el índice a borrar
            cont = 0
            # Se recorre la lista de elementos que se quieren eliminar
            for i in list_dup:
                # Se elimina el objeto indeseado
                del objects[i-cont]
                cont = cont+1
            # Una vez todos los objetos están guardados se envía la lista con todos ellos por ROS
            # Guardamos los datos en un mensaje de ROS2 para poder enviarlo posteriormente
            for i in objects:
                # Mensaje con cada detección de un objeto
                msg_aux = Detection()
                msg_aux.class_id = int(i[0])
                msg_aux.bbox.center.x = float(i[1])
                msg_aux.bbox.center.y = float(i[2])
                msg_aux.bbox.size.w = float(i[3])
                msg_aux.bbox.size.h = float(i[4])
                msg_aux.conf = float(i[5])
                
                msg_array.detections.append(msg_aux)
                
            # Se leen las imágenes generadas en la detección para enviarlas por el topic
            rgb_img = cv2.imread("/home/isa/ros2_pedro/TFG/RGB_network/image0.jpg")
            tir_img = cv2.imread("/home/isa/ros2_pedro/TFG/TIR_network/image0.jpg")
            rgb_img = node.img_conv(rgb_img)
            tir_img = node.img_conv(tir_img)
            
            # Se publican tanto las anotaciones resultantes como las imágenes resultantes de cada red
            node.publish_callback(msg_array, rgb_img, tir_img) # Se ejecuta cada 0.5 seg
            annotation_path = "/home/isa/TFG/annotation1.txt" 
            node.write_annotation(objects, annotation_path)
            
            # Borro la carpeta TFG para que se vuelva a generar en la siguiente iteración
            shutil.rmtree("/home/isa/ros2_pedro/TFG")

        else:
            time.sleep(2)




    # Cierre del nodo
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()




        

