import shutil
import sys
from ultralytics import YOLO
import cv2
import numpy as np
import os
import re


def crop_image(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)

    # Se establecen las dos esquinas del recorte
    points = [(448, 469), (840, 169)]

    # Obtener las coordenadas de los dos puntos seleccionados
    (x1, y1), (x2, y2) = points

    # Recortar la imagen utilizando los dos puntos seleccionados
    cropped_image = image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

    # Redimensionar el recorte al tamaño de la imagen térmica
    term_size = (704, 576)
    resized_image = cv2.resize(cropped_image, term_size)

    # Guardar la imagen recortada en un archivo .jpg
    # cv2.imwrite("/home/isa/TFG/recortada.jpg", resized_image)

    # Cerrar todas las ventanas una vez acaba
    # cv2.destroyAllWindows()

    return resized_image

############################################################################################


# Función donde se leen las anotaciones de una red y se devuelven los objetos detectados
def read_annotations(img, annotations_file):
    # Obtener las dimensiones de la imagen
    height, width, _ = img.shape

    
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
def check_intersection(rgb_object, index, term_objects, op=60):
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
                # print(f"Objeto {index+1} y Objeto {j+1}: {overlap_percentage}% de superposición")
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


##########################################################################################


# Función para comprobar si un mismo objeto se ha detectado con varias bboxes
def check_objects(objects):
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
def write_annotation(objects, output_path):

    # Escribir cada objeto de la lista en una línea del archivo
    with open(output_path, "w") as file:
        for obj in objects:
            # " ".join() une todo lo que haya dentro del paréntesis con un espacio
            linea = " ".join(str(element) for element in obj)  # Convertir cada elemento a cadena y unirlos con espacio
            file.write(linea + "\n")  # Escribir la línea en el archivo y agregar un salto de línea

    print("Archivo creado exitosamente.")

##########################################################################################

def draw_bounding_boxes(img, annotations_file, output_path):
    
    # Obtener las dimensiones de la imagen
    height, width, _ = img.shape
    
    # Declaramos las clases con sus índices correspondientes
    classes = {
        0: 'emergency vehicle',
        1: 'non-emergency vehicle',
        2: 'first responder',
        3: 'non-first responder'
    }
    
    # Definir los colores para cada clase
    colors = [(73, 164, 73), (183, 56, 56), (0, 0, 255), (196, 196, 10)]

    # Leer las anotaciones desde el archivo de anotaciones YOLO
    with open(annotations_file, 'r') as f:
        annotations = f.read().splitlines()
    
    # Lista para almacenar la ubicación y el área de cada objeto detectado
    detected_objects = []

    for annotation in annotations:

        # Obtener la información de cada anotación
        class_id, x_center, y_center, bbox_width, bbox_height, conf = map(float, annotation.split())
        # Se controla que el objeto no sea nulo (class_id =5)
        if class_id  != 5:
            conf = round(conf, 2) # Se redondea la confianza a 2 decimales

            # Calcular las coordenadas del bounding box
            x = int((x_center - bbox_width/2) * width) # Se multiplica porque las coordenadas del bbox son relativas al ancho y alto del mismo
            y = int((y_center - bbox_height/2) * height)
            w = int(bbox_width * width)
            h = int(bbox_height * height)
            
            # Almacenar al ubicación y el área del objeto detectado en la lista
            detected_objects.append(((x, y, w, h), class_id))

            # Se obtiene el color correspondiente a cada clase
            color = colors[int(class_id)]
            
            # Dibujar el bounding box en la imagen
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

            # Obtener el texto y el color para el texto
            text = classes[int(class_id)] + " " + str(conf)
            text_color = (255, 255, 255)  # Color blanco

            # Calcular el tamaño del texto
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

            # Calcular las coordenadas para el rectángulo del texto
            rect_x = x
            rect_y = y - text_size[1] - 10
            rect_width = text_size[0] + 10
            rect_height = text_size[1] + 10

            # Dibujar el rectángulo del texto
            cv2.rectangle(img, (rect_x, rect_y), (rect_x+rect_width, rect_y+rect_height), color, -1)

            # Dibujar el texto con el color correspondiente
            cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    # Guardamos la imagen resultante con los boundig boxes
    cv2.imwrite(output_path, img)

##########################################################################################

















# Path para la imagen rgb: /home/dgx/datasets/isa/rescate/raw/bag/extracted_dataset_2020_2021/2021/2021-06-18-12-49-08/image_right/image_right-1624013356.966232061.jpg
# Path para la imagen térmica: /home/dgx/datasets/isa/rescate/raw/bag/extracted_dataset_2020_2021/2021/2021-06-18-12-49-08/image_the/image_the-1624013357.859271049.jpg

## Si se quieren introducir argumentos desde la línea de comandos utilizar lo siguiente:
# image_rgb_path = sys.argv[1]
# image_rgb = cv2.imread(image_rgb_path)
# image_term_path = sys.argv[2]
# image_term = cv2.imread(image_term_path)

# Cargar los modelos entrenados
model_rgb = YOLO("/home/isa/TFG/results/train_rgb/weights/best.pt") # Ruta del archivo .pt obtenido tras el entrenamiento
model_term = YOLO("/home/isa/TFG/results/train_term/weights/best.pt")

# Obtener la ruta de la carpeta
carpeta_rgb = "/isa/data/rosbag_pedro/2023/images/decision_tree/RGB/"
carpeta_tir = "/isa/data/rosbag_pedro/2023/images/decision_tree/TIR/"
carpeta_anotations = "/isa/data/rosbag_pedro/2023/images/decision_tree/anotations/"

# Obtener la lista de archivos de la carpeta
archivos_rgb = os.listdir(carpeta_rgb)
archivos_tir = os.listdir(carpeta_tir)

# Definir una función para extraer números del nombre del archivo
def obtener_numeros(nombre):
    return [int(numero) if numero.isdigit() else numero for numero in re.findall(r'\d+|\D+', nombre)]

# Ordenar la lista de archivos alfanuméricamente
images_rgb = sorted([archivo for archivo in archivos_rgb if archivo.endswith(".jpg")], key=obtener_numeros)
images_tir = sorted([archivo for archivo in archivos_tir if archivo.endswith(".jpg")], key=obtener_numeros)

# Declaramos el índice de las imágenes
img_index = 0

for img in images_rgb:
    # Cargar la imagen rgb
    image_rgb_path = os.path.join(carpeta_rgb, img)  # Ruta de la imagen rgb a procesar
    image_rgb = crop_image(image_rgb_path)
    
    if img_index >= 0:    
        try:
            # Cargar la imagen térmica
            image_term_path = os.path.join(carpeta_tir, images_tir[img_index])  # Ruta de la imagen térmica a procesar
            image_term = cv2.imread(image_term_path)
        except:
            # No hay más imágenes térmicas
            break
        
        # Se borra la carpeta /home/isa/ultralytics/TFG con las anotaciones ya leidas
        try:
            shutil.rmtree("/home/isa/ultralytics/TFG")
        except:
            pass
        
        # Realizar las detecciones en las imagenes
        results_rgb = model_rgb.predict(project="TFG", name="RGB_network", stream=False, source=image_rgb, save=True, save_txt=True, save_conf=True)
        results_term = model_term.predict(project="TFG", name="TIR_network", stream=False, source=image_term, save=True, save_txt=True, save_conf=True)
        # save=True guarda la imagen resultado en ultralytics/runs/detect/predict2
        # save_txt=True guarda el resultado como un archivo txt
        # save_conf=True guarda el resultado con los valores de confianza de cada predicción
        # stream=True ahorra memoria para recibir muchas imágenes y dar resultados más rápidos


        ### A continuación, se crea el algoritmo de la fusión

        # Leemos las predicciones hechas por las redes
        objects_rgb = read_annotations(image_rgb, "/home/isa/ultralytics/TFG/RGB_network/labels/image0.txt")
        objects_term = read_annotations(image_term, "/home/isa/ultralytics/TFG/TIR_network/labels/image0.txt")
        
        # Copiamos las imágenes resultado de las redes a otra carpeta
        src_rgb = "/home/isa/ultralytics/TFG/RGB_network/image0.jpg"
        dst_rgb = "/isa/data/rosbag_pedro/2023/images/final_results/RGB/img" + str(img_index) + ".jpg"
        src_tir = "/home/isa/ultralytics/TFG/TIR_network/image0.jpg"
        dst_tir = "/isa/data/rosbag_pedro/2023/images/final_results/TIR/img" + str(img_index) + ".jpg"

        shutil.copy(src_rgb, dst_rgb)
        shutil.copy(src_tir, dst_tir)

        # Declaramos las clases con sus índices correspondientes
        classes = {
            0: 'emergency vehicle',
            1: 'non-emergency vehicle',
            2: 'first responder',
            3: 'non-first responder',
            5: '' # Vacío, ninguna detección encontrada
        }

        # Inicializamos las coordenadas resultado en todo 0
        (x, y, w, h, prob) = (0, 0, 0, 0, 0)

        # Se obtiene el ancho y largo de las imagenes (tienen las mismas dimensiones) para posteriores conversiones
        height, width, _ = image_rgb.shape

        # Lista con los objetos térmicos detectados en ambas redes
        shared_term_obj = []

        # Lista con las detecciones finales
        objects = []

        # El objeto viene representado así: ((x, y, w, h), class_id)
        # Identificamos los objetos detectados por ambas redes y separamos cada tipo de detección
        for i, object_rgb in enumerate(objects_rgb):
            match_object = check_intersection(object_rgb, i, objects_term)
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
        list_dup = check_objects(objects)
        list_dup.sort()
        # Se crea un contador para controlar el índice a borrar
        cont = 0
        # Se recorre la lista de elementos que se quieren eliminar
        for i in list_dup:
            # Se elimina el objeto indeseado
            del objects[i-cont]
            cont = cont+1

        # Una vez todos los objetos están guardados se escriben en un archivo .txt
        annotation_path = "/home/isa/TFG/annotation1.txt"
        write_annotation(objects, annotation_path)
            
        # Finalmente se lee este archivo .txt y se dibujan los bbox en la imagen
        output_path = "/isa/data/rosbag_pedro/2023/images/final_results/fusion/img" + str(img_index) + ".jpg"
        draw_bounding_boxes(image_rgb, annotation_path, output_path)
    
    img_index += 1
        

