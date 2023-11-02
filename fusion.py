import sys
from ultralytics import YOLO
import cv2
import numpy as np


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
    # cv2.imwrite("/root/TFG/recortada.jpg", resized_image)

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

############################################################################################

# Función para checkear la interseccion de dos bounding boxes de ambas redes
def check_intersection(rgb_object, index, term_objects):
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
        
            if overlap_percentage >= 70:
                cont = cont + 1
                # Mostrar el porcentaje de superposición y guardarlo
                print(f"Objeto {index+1} y Objeto {j+1}: {overlap_percentage}% de superposición")
                # Si varios objetos tienen un IoU > 70% se escoge el mayor
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

# Función para obtener la iluminación media de la imágen rgb
def get_brightness(image):
    # Convertir el frame a escala de grises
    gray_frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Calcular el histograma
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])

    # Normalizar el histograma
    hist_norm = hist / np.sum(hist)

    # Obtenemos la media del histograma normalizado para saber la intensidad de la imagen
    intensity_mean = np.mean(hist_norm)

    return intensity_mean

##########################################################################################

# Función para crear la tabla de probabilidad

"""
        |   RGB     |   NO RGB    |     
-------------------------------------------------
TERM    |     a     |      b      |   term_prob
-------------------------------------------------
NO TERM |     c     |      d      | no_term_prob
-------------------------------------------------
        |  rgb_prob | no_rgb_prob |     

"""

def prob_table(conf_rgb, conf_term, mode):
    term_prob = conf_term
    no_term_prob = 1-conf_term
    rgb_prob = conf_rgb
    no_rgb_prob = 1-conf_rgb
    output = -1
    # mode = 1 es la tabla para ambas detecciones iguales
    if mode == 1:
        # Obtenemos las probabilidades de los extremos de las redes iniciales
        # Damos un valor inicial al hueco d (el menos probable en este caso)
        d = 0.01
        b = no_rgb_prob - d
        a = term_prob - b
        c = rgb_prob - a
        # Solamente se usará uno de los valores de la tabla así que crearemos una variable output
        output = a # Probabilidad de que acierten ambas redes
    # mode = 2 es la tabla para solo detecta la red rgb 
    elif mode == 2:
        # Obtenemos las probabilidades de los extremos de las redes iniciales
        # En este caso la probabilidad de que detecte la red térmica será de 0.1
        # Damos un valor inicial al hueco b (el menos probable en este caso)
        b = 0.01
        d = no_rgb_prob - b
        a = term_prob - b
        c = rgb_prob - a
        # Solamente se usará uno de los valores de la tabla así que crearemos una variable output
        output = c # Probabilidad de que acierte solo la red rgb
    # mode = 3 es la tabla para solo detecta la red térmica 
    elif mode == 3:
        # Obtenemos las probabilidades de los extremos de las redes iniciales
        # En este caso la probabilidad de que detecte la red rgb será de 0.1
        # Damos un valor inicial al hueco c (el menos probable en este caso)
        c = 0.01
        d = no_term_prob - c
        a = rgb_prob - c
        b = term_prob - a
        # Solamente se usará uno de los valores de la tabla así que crearemos una variable output
        output = b # Probabilidad de que acierte solo la red térmica
        print("El valor obtenido para detección en la térmica pero no en la rgb es " + str(output))
    else:
        print("Invalid mode, please check the choice")
    
    return output

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
model_rgb = YOLO("/root/TFG/results/train_rgb/weights/best.pt") # Ruta del archivo .pt obtenido tras el entrenamiento
model_term = YOLO("/root/TFG/results/train_term/weights/best.pt")

# Cargar la imagen rgb
image_rgb_path = "/root/TFG/image_right-1624013356.966232061.jpg"  # Ruta de la imagen rgb a procesar
image_rgb = crop_image(image_rgb_path)

# Cargar la imagen térmica
image_term_path = "/root/TFG/image_the-1624013357.859271049.jpg"  # Ruta de la imagen térmica a procesar
image_term = cv2.imread(image_term_path)

# Realizar las detecciones en las imagenes
results_rgb = model_rgb.predict(stream=True, source=image_rgb, save=True, save_txt=True, save_conf=True)
results_term = model_term.predict(stream=True, source=image_term, save=True, save_txt=True, save_conf=True)
# save=True guarda la imagen resultado en ultralytics/runs/detect/predict2
# save_txt=True guarda el resultado como un archivo txt
# save_conf=True guarda el resultado con los valores de confianza de cada predicción
# stream=True ahorra memoria para recibir muchas imágenes y dar resultados más rápidos


### A continuación, se crea el algoritmo de la fusión

# Leemos las predicciones hechas por las redes
objects_rgb = read_annotations(image_rgb, "/root/ultralytics/runs/detect/predict/labels/image0.txt")
objects_term = read_annotations(image_term, "/root/ultralytics/runs/detect/predict2/labels/image0.txt")

# Declaramos las clases con sus índices correspondientes
classes = {
    0: 'emergency vehicle',
    1: 'non-emergency vehicle',
    2: 'first responder',
    3: 'non-first responder'
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
        # Ahora separamos si coinciden en la clase raíz
        if (class_rgb < 2 and class_term < 2) or (class_rgb > 1 and class_term > 1):
            # Ahora separamos si coinciden en la clase específica
            if class_rgb == class_term:
                # Calculamos la probabilidad de que ambas redes hayan acertado
                prob = prob_table(conf_rgb, conf_term, 1)
                print("Probabilidad obtenida en ambas detectan el mismo objeto es " + str(prob))

                # Escogemos el bbox con mayor confianza inicial
                ob_class = class_rgb
                if conf_rgb >= conf_term:
                    (x, y, w, h) = (xc1/width, yc1/height, w1/width, h1/height)
                else:
                    (x, y, w, h) = (xc2/width, yc2/height, w2/width, h2/height)
        
        # Aquí se incluyen dos posibles casos con el mismo procedimiento:
        # - Coinciden en clase raíz pero no en clase específica
        # - No coinciden en clase raíz. Este caso no debería ocurrir pero se incluye por seguridad
        else:
            # Calculamos la probabilidad de acierto de cada red basándonos en la iluminación
            brightness = get_brightness(image_rgb)
            # Cuando la iluminación media de la imagen es inferior a 0.3 se le da mayor 
            # probabilidad a la red térmica 
            if brightness <= 0.3:
                prob_rgb_and_brightness = 0.3
                prob_term_and_brightness = 0.7
            # Cuando la iluminación media de la imagen es superior a 0.3 se le da mayor 
            # probabilidad a la red rgb 
            else:
                prob_rgb_and_brightness = 0.7
                prob_term_and_brightness = 0.3
            # Usamos la regla del producto de probabilidad
            prob_rgb = prob_rgb_and_brightness / conf_rgb
            prob_term = prob_term_and_brightness / conf_term
            # Escogemos el bbox y la probabilidad final del objeto con mayor confianza
            if prob_rgb > prob_term:
                prob = prob_rgb
                (x, y, w, h) = (xc1/width, yc1/height, w1/width, h1/height)
                ob_class = class_rgb
            else:
                prob = prob_term
                (x, y, w, h) = (xc2/width, yc2/height, w2/width, h2/height)
                ob_class = class_term
            print("Probabilidad obtenida en mismo objeto pero distinta clase es " + str(prob))
            
            
    # Detecta objeto red rgb y no la red térmica
    else:
        # Cuando no se detecta el objeto con la red térmica se utiliza conf_term = 0.1
        conf_term = 0.1
        prob_RynoT = prob_table(conf_rgb, conf_term, 2)
        # Se multiplica esta probabilidad por un factor de seguridad para que no se supere 
        # la probabilidad mayor que 1
        
        # Una vez tengo esta probabilidad creo la probabilidad final condicionando la 
        # detección con la iluminación
        brightness = get_brightness(image_rgb)
        # Cuando la iluminación media de la imagen es superior a 0.3 se le da mayor 
        # probabilidad a la red rgb 
        if brightness > 0.3:
            prob_rgb_and_brightness = 0.7
        else:
            prob_rgb_and_brightness = 0.3
        # Usamos la regla del producto de probabilidad
        prob = prob_rgb_and_brightness / prob_RynoT
        print("Probabilidad obtenida en solo detecta objeto rgb es " + str(prob))
        # Ahora se podría condicionar para coger este objeto dependiendo de si supera un 
        # cierto umbral pero en principio no se va restringir de ningún modo

        # Guardamos las coordenadas y dimensiones del bbox
        (x, y, w, h) = (xc1/width, yc1/height, w1/width, h1/height)
        ob_class = class_rgb

    objects.append([ob_class, x, y, w, h, prob])
    

# Ahora recogemos el caso en que el objeto ha sido detectado por la red térmica pero no por la rgb
for i, object_term in enumerate(objects_term):
    if i not in shared_term_obj:
        # Extraemos los datos del objeto térmico
        (x2, y2, w2, h2), ob_class, conf_term = object_term
        # Reconvertimos las coordenadas al centro del bbox térmico
        xc2 = x2 + w2/2
        yc2 = y2 + h2/2
        # Cuando no se detecta el objeto con la red rgb se utiliza conf_rgb = 0.1
        conf_rgb = 0.1
        prob_TynoR = prob_table(conf_rgb, conf_term, 3)
        # Se multiplica esta probabilidad por un factor de seguridad para que no se supere 
        # la probabilidad mayor que 1
        
        # Una vez tengo esta probabilidad creo la probabilidad final condicionando la 
        # detección con la iluminación
        brightness = get_brightness(image_rgb)
        print("La iluminación en la imagen es de " + str(brightness))
        # Cuando la iluminación media de la imagen es superior a 0.3 se le da mayor 
        # probabilidad a la red rgb 
        if brightness > 0.3:
            prob_term_and_brightness = 0.3
        else:
            prob_term_and_brightness = 0.7
        # Usamos la regla del producto de probabilidad
        prob = prob_term_and_brightness / prob_TynoR
        print("Probabilidad obtenida en solo detecta objeto térmico es " + str(prob))
        # Ahora se podría condicionar para coger este objeto dependiendo de si supera un 
        # cierto umbral pero en principio no se va restringir de ningún modo
        # Por último, sacamos los parámetros del objeto
        (x, y, w, h) = (xc2/width, yc2/height, w2/width, h2/height)
        ob_class = class_term
        
        # Se guarda el objeto en una variable
        objects.append([ob_class, x, y, w, h, prob])


# Una vez todos los objetos están guardados se escriben en un archivo .txt
annotation_path = "/root/TFG/annotation1.txt"
write_annotation(objects, annotation_path)
    
# Finalmente se lee este archivo .txt y se dibujan los bbox en la imagen
output_path = "/root/TFG/image_pred2.jpg"
draw_bounding_boxes(image_rgb, annotation_path, output_path)
        

