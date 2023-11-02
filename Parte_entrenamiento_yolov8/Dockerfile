FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Downloads to user config dir
ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

# Actualizo el sistema e instalo dependencias
RUN apt-get update \
    && apt install --no-install-recommends -y gcc git zip curl htop \
    libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg g++ python3-pip

# Instalo otras dependencias (necesario para la red)
RUN pip install opencv-python matplotlib thop pandas seaborn

# Actualizaciones de seguridad
RUN apt upgrade --no-install-recommends -y openssl tar

# Creo el directorio de trabajo
RUN mkdir -p /usr/src/ultralytics
WORKDIR /root

# Descargo el repositorio de ultralytics (librería de la red neuronal)
RUN git clone https://github.com/ultralytics/ultralytics /root/ultralytics
# Copio los pesos de la red neuronal a utilizar (como quiero usar la red s, m o l deberé cambiar el yolov8n.pt)
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt /root/ultralytics/model/

# Instalo clearml para monitorizar los resultados
RUN pip install clearml
# Ejecutar el siguiente comando en la terminal donde ejecute el entrenamiento
# RUN clearml-init 
# Ahora copio y pego el interior del archivo clearml_credentials.py 

# Copio el archivo main.py para poner a entrenar la red de la carpeta TFG a la carpeta ultralytics
# Si se modifica el main.py después de haber creado la imágen habrá que borrar la copia y volverlo a copiar
# Tengo que hacer este comando a mano dentro del contenedor porque cuando se crea la imagen todavía no se ha creado el volumen y no puedo copiarlo
# RUN cp /root/TFG/main.py /root/ultralytics

# Creo un volumen con la carpeta para el entrenamiento
VOLUME root/TFG 
# Esta carpeta es donde se copiará el volumen (path del contenedor)

# Establezco un usuario normal en vez de superusuario
# USER pedrop



