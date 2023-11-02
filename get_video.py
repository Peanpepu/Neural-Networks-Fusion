import shutil
import os
from ultralytics import YOLO
import cv2
import numpy as np
import threading
import time


def crop_image(image):

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

    # Leer las anotaciones desde el archivo de anotaciones YOLO
    with open(annotations_file, 'r') as f:
        annotations = f.read().splitlines()
    
    # Lista para almacenar la ubicación y el área de cada objeto detectado
    detected_objects = []

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
    return detected_objects

##############################################################################################

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
    
    # Devolvemos la imagen resultante con los boundig boxes
    return img

##########################################################################################















# Path para la imagen rgb: /home/dgx/datasets/isa/rescate/raw/bag/extracted_dataset_2020_2021/2021/2021-06-18-12-49-08/image_right/image_right-1624013356.966232061.jpg
# Path para la imagen térmica: /home/dgx/datasets/isa/rescate/raw/bag/extracted_dataset_2020_2021/2021/2021-06-18-12-49-08/image_the/image_the-1624013357.859271049.jpg

# Cargar los modelos entrenados
model_rgb = YOLO("/home/isa/TFG/results/train_rgb/weights/best.pt") # Ruta del archivo .pt obtenido tras el entrenamiento
model_term = YOLO("/home/isa/TFG/results/train_term/weights/best.pt")

# Creo una variable frame para igualar los frames de la cámara rgb y la térmica
f_term = 0

# Especificar dirección del vídeo rgb
video_rgb_path = "/home/isa/TFG/right_video.mp4"  # Ruta de la imagen rgb a procesar
# Abrir vídeo rgb
cap_rgb = cv2.VideoCapture(video_rgb_path)

# Especificar dirección del vídeo térmico
video_term_path = "/home/isa/TFG/the_video.mp4"  # Ruta de la imagen rgb a procesar
# Abrir vídeo térmico
cap_term = cv2.VideoCapture(video_term_path)

# Se crea una pestaña para mostrar el vídeo
# cv2.namedWindow("Red fusionada")
# thread = threading.Thread(target=show_img)

# Se crea un vídeo para rellenar con el frame de cada iteración
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
test_size = tuple(reversed(cap_term.read()[1].shape[:2]))
video_writer_rgb = cv2.VideoWriter("/home/isa/TFG/video_rgb.mp4", fourcc, 5, test_size, True)
video_writer_term = cv2.VideoWriter("/home/isa/TFG/video_term.mp4", fourcc, 5, test_size, True)
# fourcc es el formato de vídeo que voy a usar
# 25 son los frames por segundo que voy a usar
# (704, 576) es el tamaño de cada frame
# True indica si la imagen es a color

cont2=0
# Se itera por todos los frames del vídeo
while cap_rgb.isOpened() and cap_term.isOpened():

    # Contador para comprobar 50 imágenes
    cont2=cont2+1
    if cont2 > 536:
        break
    print("Contador " + str(cont2))
    
    # Se lee el frame del vídeo rgb
    ret1, frame_rgb = cap_rgb.read()
    # ret1 es un booleano que dice si se ha podido leer el frame o no
    image_rgb = crop_image(frame_rgb)
    
    # Se lee el frame del vídeo térmico
    while(f_term % 5 != 0):
        f_term = f_term+1
        ret2, image_term = cap_term.read()
    if(f_term % 100 == 0):
        f_term = f_term+1
        ret2, image_term = cap_term.read()
        while(f_term % 5 != 0):
            f_term = f_term+1
            ret2, image_term = cap_term.read()
    print(str(ret1) + " " + str(ret2)) 
    # Si alguno de los vídeos se queda sin frames se cierra el vídeo
    if not ret1 or not ret2:
        break

    # Realizar las detecciones en las imagenes
    results_rgb = model_rgb.predict(project="TFG", name="RGB_network", stream=False, source=image_rgb, save=True, save_txt=False, save_conf=False)
    results_term = model_term.predict(project="TFG", name="TIR_network", stream=False, source=image_term, save=True, save_txt=False, save_conf=False)
    # save=True guarda la imagen resultado en ultralytics/runs/detect/predict2
    # save_txt=True guarda el resultado como un archivo txt
    # save_conf=True guarda el resultado con los valores de confianza de cada predicción
    # stream=True ahorra memoria para recibir muchas imágenes y dar resultados más rápidos
    
    # Declaramos las clases con sus índices correspondientes
    classes = {
        0: 'emergency vehicle',
        1: 'non-emergency vehicle',
        2: 'first responder',
        3: 'non-first responder'
    }

        
    # Finalmente se lee este archivo .txt y se dibujan los bbox en la imagen
    # output_path = "/home/isa/TFG/image_pred3.jpg"
    output_path_rgb = "/home/isa/TFG/images_res/rgb/image" + str(cont2) + ".jpg"
    output_path_term = "/home/isa/TFG/images_res/term/image" + str(cont2) + ".jpg"
    img_rgb = cv2.imread("/home/isa/ros2_pedro/src/ultralytics/ultralytics/TFG/RGB_network/image0.jpg")
    img_term = cv2.imread("/home/isa/ros2_pedro/src/ultralytics/ultralytics/TFG/TIR_network/image0.jpg")
    cv2.imwrite(output_path_rgb, img_rgb)
    cv2.imwrite(output_path_term, img_term)
    # Mostramos la imagen y esperamos entrada de teclado, si escape se sale del bucle
    # show_img()
    # if cv2.waitKey(5) == 27:
    #     break
    
    # Se guarda la imagen en el vídeo que se quiere crear
    video_writer_rgb.write(img_rgb)
    video_writer_term.write(img_term)
    
    # Se eliminan las anotaciones indeseadas
    shutil.rmtree("/home/isa/ros2_pedro/src/ultralytics/ultralytics/TFG")
    

# Se cierra el vídeo
video_writer_rgb.release()
video_writer_term.release()


        

