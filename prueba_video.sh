# Launcher para borrar carpetas necesarias, ejecutar fusion.py y 
# copiar resultados a la carpeta TFG (desde carpeta ~/ultralytics)

# Elimino carpetas y archivos
rm -r ./TFG
rm ~/TFG/annotation1.txt
rm ./video_fusion.py

# Copio fusion.py
cp ~/TFG/video_fusion.py .

# Elimino el vídeo anterior
rm ~/TFG/video_fusion.mp4

# Ejecuto el algoritmo
python3 video_fusion.py


