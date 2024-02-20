# Autor: Jorge Caicedo
# Fecha: 17/01/2024
# Este script realiza la separación de los conjuntos de entrenamiento
# validación y prueba con los porcentajes 70, 15 y 15 respectivamente.

# Importe de bibliotecas
import os
import glob
import random
import shutil

# Directorio de imágenes
data_path = "D:\\roses_diameter\\dataset\\yolo_ds"

# Directorios de almacenamiento de las imágenes
train_folder = os.path.join(data_path, 'train')
test_folder = os.path.join(data_path, 'test')
val_folder = os.path.join(data_path, 'valid')

# Especificación de las extensiones de archivo
image_extensions = ['.jpg', '.jpeg', '.png']

# Crear una lista de nombres de archivos existentes en data_path
imgs_list = [filename for filename in os.listdir(data_path) if os.path.splitext(filename)[-1] in image_extensions]

# Semilla para reproductividad
random.seed(0)

# Aleatorizar la lista de nombres de archivos
random.shuffle(imgs_list)

# Cálculo del número de imágenes para cada directorio
train_size = int(len(imgs_list) * 0.70)
val_size = int(len(imgs_list) * 0.15)
test_size = int(len(imgs_list) * 0.15)

# Crear directorios si no existen
for folder_path in [train_folder, val_folder, test_folder]:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Copiado de las imágenes a sus directorios correspondientes
for i, f in enumerate(imgs_list):
    if i < train_size:
        dest_folder = train_folder
    elif i < train_size + val_size:
        dest_folder = val_folder
    else:
        dest_folder = test_folder
    shutil.copy(os.path.join(data_path, f), os.path.join(dest_folder, f))