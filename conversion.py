import os
import xml.etree.ElementTree as ET

def convert_xml_to_yolo(xml_file_path):
    # Parse el archivo XML
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Obtener el nombre del archivo sin extensión
    file_name = os.path.splitext(os.path.basename(xml_file_path))[0]

    output_dir = "/home/pedrop/Desktop/TFG/data/dataset_rgb/labels/test/"
    output_file_path = os.path.join(output_dir, file_name + ".txt")

    classes = ["emergency vehicle", "non-emergency vehicle","first responder", "non-first responder"]

    # Crear el archivo TXT
    with open(output_file_path, "w") as f:
        # Iterar a través de todos los objetos en el archivo XML
        for obj in root.findall('object'):
            # Obtener el nombre del objeto (clase) y su índice
            if obj.find('name').text in classes:
                name = classes.index(obj.find('name').text)
            else:
                name = 4 # Si la clase no corresponde con las establecidas clase = 4 (no existe)
                print("Error in classes name in file " + xml_file_path)         

            # Obtener los límites del bounding box
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # Calcular el centro, la anchura y la altura del bounding box
            x = (xmin + xmax) / 2 / float(root.find('size').find('width').text)
            y = (ymin + ymax) / 2 / float(root.find('size').find('height').text)
            w = (xmax - xmin) / float(root.find('size').find('width').text)
            h = (ymax - ymin) / float(root.find('size').find('height').text)

            # Escribir la información en el archivo TXT
            f.write(f"{name} {x} {y} {w} {h}\n")

# Obtener la lista de todos los archivos XML en el directorio
#"/home/pedrop/Desktop/TFG/data/dataset_term/test/labels"
# Para las anotaciones de test de térmica el path es el siguiente:
# /home/pedrop/Desktop/TFG/data/dataset_term/test/labels
# Para las anotaciones de train de térmica el path es el siguiente:
xml_dir =  "/home/pedrop/Desktop/TFG/data/dataset_rgb/test/labels/"
# Para las anotaciones de test de rgb el path es el siguiente:
# /home/pedrop/Desktop/TFG/data/dataset_rgb/test/labels
# Para las anotaciones de train de rgb el path es el siguiente:
# /home/pedrop/Desktop/TFG/data/dataset_rgb/train/labels
xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]

# Convertir cada archivo XML a YOLO
for xml_file in xml_files:
    xml_file_path = os.path.join(xml_dir, xml_file)
    convert_xml_to_yolo(xml_file_path)
