import cv2


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
    cv2.imwrite("/isa/data/rosbag_pedro/2023/imgs_prueba_final/recortada.jpg", resized_image)

    # Cerrar todas las ventanas una vez acaba
    cv2.destroyAllWindows()


# Ejemplo de uso
# image_path = "C:\Users\pedro\Desktop\TFG\Mi_TFG\image_right-1624013356.966232061.jpg"  # Ruta de la imagen que deseas recortar
# crop_image(image_path)


# El tamaño de la imagen original es el siguiente: altura = 720 y ancho = 1280
# El tamaño de la imagen recortada es el siguiente: altura = 300 y ancho = 392
# La esquina inferior izquierda se encuentra en (x, y) = (448, 469)
# La esquina superior derecha se encuentra en (x, y) = (840, 169)