import os 
import time

from ultralytics import YOLO
import cv2
import numpy as np

VIDEOS_DIR = os.path.join('.', 'videos')
video_path = os.path.join(VIDEOS_DIR, 'video_para_probar.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
# H, W, _ = frame.shape
# out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*"MP4V"), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path_rgb = os.path.join(".", "models", "red_entrenada.pt")
model_path_therm = os.path.join(".", "models", "red_entrenada.pt")


# Load a model
model_rgb = YOLO(model_path_rgb) # load rgb model
model_therm = YOLO(model_path_therm) # load thermical model
threshold = 0.5
class_name_dict = {0: "emergency vehicle", 1: "non-emergency vehicle", 2: "first responder", 3: "non-first responder"} # Classes names

while ret:
    # Convertir el frame a escala de grises
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcular el histograma
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])

    # Normalizar el histograma
    hist_norm = hist / np.sum(hist)

    # Obtenemos la media del histograma normalizado para saber la intensidad de la imagen
    intensity_mean = np.mean(hist_norm)

    if intensity_mean < 0.3: # Si hay poca intensidad de luz en la imagen
        # La red térmica tiene más peso
        model = model_therm
    else: # Si hay suficiente intensidad de luz en la imagen
        # La red RGB tiene más peso
        model = model_rgb
    results = model(frame)[0]