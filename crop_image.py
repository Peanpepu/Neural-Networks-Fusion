# Código para recortar la imagen rgb al tamaño de la térmica a ojo. De aquí obtengo las medidas para hacer el 
# recorte automático de cada imagen rgb

import cv2

def crop_image(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)

    # Mostrar la imagen y permitir que el usuario seleccione dos puntos
    clone = image.copy()
    points = []

    # Obtener el tamaño de la imagen original y mostrarlo
    height, width = image.shape[:2]
    print("El tamaño de la imagen original es el siguiente: altura = " + str(height) + " y ancho = " + str(width))

    def select_points(event, x, y, flags, param):
        nonlocal points

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

            # Dibujar un círculo en el punto seleccionado
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)

            # Mostrar los puntos seleccionados
            for i, (px, py) in enumerate(points):
                cv2.putText(clone, f"({px}, {py})", (px+10, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(clone, (px, py), 2, (0, 255, 0), -1)

            # Actualizar la ventana de visualización
            cv2.imshow("Image", clone)

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", select_points)

    # Esperar a que el usuario seleccione dos puntos
    while len(points) < 2:
        cv2.imshow("Image", clone)
        cv2.waitKey(1)

    # Cerrar la ventana de visualización
    cv2.destroyAllWindows()

    # Obtener las coordenadas de los dos puntos seleccionados
    (x1, y1), (x2, y2) = points

    # Recortar la imagen utilizando los dos puntos seleccionados
    cropped_image = image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

    # Redimensionar el recorte al tamaño de la imagen térmica
    term_size = (704, 576)
    resized_image = cv2.resize(cropped_image, term_size)

    # Guardar la imagen recortada en un archivo .jpg
    cv2.imwrite("/isa/data/rosbag_pedro/2023/imgs_prueba_final/recortada.jpg", resized_image)

    # Mostrar la imagen recortada
    cv2.imshow("Cropped Image", resized_image)
    cv2.waitKey(0) # Espera hasta que se pulsa una tecla para cerrarse

    # Obtener el tamaño de la imagen original y mostrarlo
    height_r, width_r = cropped_image.shape[:2]
    print("El tamaño de la imagen recortada es el siguiente: altura = " + str(height_r) + " y ancho = " + str(width_r))

    # Obtener la ubicación de los puntos recortados
    print("La esquina inferior izquierda se encuentra en (x, y) = (" + str(x1) + ", " + str(y1) + ")")
    print("La esquina superior derecha se encuentra en (x, y) = (" + str(x2) + ", " + str(y2) + ")")

    # Cerrar todas las ventanas una vez acaba
    cv2.destroyAllWindows()


# Ejemplo de uso
image_path = "/isa/data/rosbag_pedro/2023/imgs_prueba_final/img_rgb_0.jpg"  # Ruta de la imagen que deseas recortar
# \\wsl.localhost\Ubuntu\home\pedro\TFG_images\rgb\image_right\
crop_image(image_path)


# El tamaño de la imagen original es el siguiente: altura = 720 y ancho = 1280
# El tamaño de la imagen recortada es el siguiente: altura = 300 y ancho = 392
# La esquina inferior izquierda se encuentra en (x, y) = (448, 469)
# La esquina superior derecha se encuentra en (x, y) = (840, 169)