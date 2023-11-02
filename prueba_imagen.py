from ultralytics import YOLO
import cv2
import rospy
from std_msgs.msg import String

# Path para la imagen rgb: /home/dgx/datasets/isa/rescate/raw/bag/2021/2021-06-18-13-05-09/TIR/frame0000.jpg
# Path para la imagen térmica: /home/dgx/datasets/isa/rescate/raw/bag/2021/2021-06-18-13-05-09/TIR/frame0000.jpg


# Cargar el modelo pre-entrenado
# model_path = os.path.join("/root/TFG/results/train_rgb/weights/best.pt") # Ruta del archivo .pt obtenido tras el entrenamiento
model_rgb = YOLO("/root/TFG/results/train_rgb/weights/best.pt")
model_term = YOLO("/root/TFG/results/train_term/weights/best.pt")

# Cargar la imagen rgb
image_rgb_path = "/root/TFG/data/dataset_rgb/images/test/frame2157.jpg"  # Ruta de la imagen rgb a procesar
image_rgb = cv2.imread(image_rgb_path)

# Cargar la imagen térmica
image_term_path = "/root/TFG/data/dataset_rgb/images/test/frame2157.jpg"  # Ruta de la imagen térmica a procesar
image_term = cv2.imread(image_term_path)

# Realizar las detecciones en las imagenes
results_rgb = model_rgb.predict(source=image_rgb, save=True, save_txt=True, save_conf=True)
results_term = model_term.predict(source=image_term, save=True, save_txt=True, save_conf=True)
# save=True guarda la imagen resultado en ultralytics/runs/detect/predict2
# save_txt=True guarda el resultado como un archivo txt
# save_conf=True guarda el resultado con los valores de confianza de cada predicción



# A continuación, se crea una función que dibuje los bounding boxes en la imagen original a partir del txt
def draw_bounding_boxes(image_path, annotations_file, output_path):
    # Cargar la imagen
    img = cv2.imread(image_path)
    
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
    
# Ahora llamamos a la función con ambas imágenes
annotation_path_rgb = "/root/ultralytics/runs/detect/predict/labels/frame2157.txt"
output_path_rgb = "/root/TFG/results/pruebas/pred_frame2157.jpg"
im_rgb = image_rgb_path

annotation_path_term = "/root/ultralytics/runs/detect/predict/labels/frame2157.txt"
output_path_term = "/root/TFG/results/pruebas/pred_frame2157.jpg"
im_term = image_term_path

draw_bounding_boxes(im_rgb, annotation_path_rgb, output_path_rgb)
draw_bounding_boxes(im_term, annotation_path_term, output_path_term)

