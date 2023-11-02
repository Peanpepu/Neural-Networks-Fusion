# Launcher para borrar carpetas necesarias, ejecutar fusion.py y 
# copiar resultados a la carpeta TFG (desde carpeta ~/ultralytics)

# Elimino carpetas y archivos
rm -r ./TFG
rm ~/TFG/annotation1.txt
rm ~/TFG/img_rgb.txt
rm ~/TFG/img_term.txt
rm ~/TFG/image_pred3.jpg
rm ~/TFG/img_rgb.jpg
rm ~/TFG/img_term.jpg
rm ./fusion_v2.py

# Copio fusion.py
cp ~/TFG/fusion_v2.py .

# Ejecuto el algoritmo
python3 fusion_v2.py

# Copio los archivos creados
mv ./TFG/RGB_network/labels/image0.txt ~/TFG/img_rgb.txt
mv ./TFG/RGB_network/image0.jpg ~/TFG/img_rgb.jpg
mv ./TFG/TIR_network/labels/image0.txt ~/TFG/img_term.txt
mv ./TFG/TIR_network/image0.jpg ~/TFG/img_term.jpg
