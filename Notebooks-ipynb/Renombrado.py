# Author: Jorge Caicedo
# Date: 17/01/2024
# Este script realiza el renombrado del un directorio completo
# de imágenes partiendo desde el número 1, 2, ..., y guarda 
# los resultados en el mismo directorio especificado.

# Importe de bibliotecas
import os
import glob

# Directorio de imágenes
path = "D:\\roses_diameter\\dataset\\yolo_ds"
files = glob.glob(path + '/*')

# Ciclo para renombrado de imágenes a partir del número 1
for i, f in enumerate(files, 1):
    ftitle, fext = os.path.splitext(f)
    os.rename(f, os.path.join(path,str(i)) + fext)
	
# Impresión del listado de los nuevos nombres de imágenes generado
print(glob.glob(path + '/*'))