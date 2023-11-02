# Launcher para borrar carpetas necesarias, ejecutar fusion.py y 
# copiar resultados a la carpeta TFG (desde carpeta ~/ultralytics)

# Elimino carpetas y archivos
rm -r ./TFG
rm ~/TFG/annotation1.txt
rm ./prueba_ros.py

# Copio y le doy permisos de ejecución a prueba_ros.py
cp ~/TFG/prueba_ros.py .
chmod +x prueba_ros.py

# Compilo el paquete de ros2
cd ../../..
colcon build

# Configuro el setup
. install/setup.bash

# Ejecuto el algoritmo
cd src
ros2 run ultralytics prueba_ros.py


