# Autor: Jorge Caicedo
# Fecha: 01-02-2024
# Este script implementa la lógica de progrmación necesaria para un microservicio
# que sea capaz de recibir un escuchar una solicitud POST en la que se envíe un
# archivo de imagen o video (src_file) y retorne como resultado el mismo archivo
# con la anotaciones que indiquen los resultados de medición calculados para el
# archivo dado.

# Importe de bibliotecas necesarias
import torch
import cv2
from flask import Flask, jsonify, request, send_file
from joblib import load
import numpy as np
import os
from werkzeug.utils import secure_filename
import pathlib

#Líneas implementadas para solucionar un problema de compatibilidad
# con error PosixPath 
##---------
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
#---------

# Definición de variables globales
# Configuración para especificar la carpeta de carga y permitir extensiones específicas
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','mp4'}
WEIGHTS_PATH = os.path.join("weights/",'best.pt')
TRAINED_IMG_SIZE = 1280
MIN_CONFIDENCE = 0.6
#Establecimiento del ratio de relacion pixel/centímetro
        # Se siguen las recomendaciones establecidas en la Sección 2.6.1
        #235 px = 1 cm

# Ratio_px_cm --> distancia de la cámara 15 cm
RATIO_PX_CM_15 = 235/1
# Ratio_px_cm --> distancia de la cámara 25 cm
RATIO_PX_CM_25 = 137/1

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/api/ml/img", methods=['GET', 'POST'])
def prediccion_img():
     # Verificar si la solicitud POST tiene el archivo adjunto
    if 'img_src' not in request.files:
        return jsonify({'error': 'No se encontró el archivo en la solicitud'}), 400

    file = request.files['img_src']

    # Verificar si el archivo tiene un nombre y una extensión válidos
    if file.filename == '':
        return jsonify({'error': 'El archivo no tiene un nombre válido'}), 400

    if file and allowed_file(file.filename):
        # Guardar el archivo en la carpeta de carga
        filename = secure_filename(str(file.filename))
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Predecir el díametro
        predicted_img = predecir_diametro_img(WEIGHTS_PATH,file_path,TRAINED_IMG_SIZE)

        # Guardar la imagen con las predicciones en el directorio correspondiente
        predicted_img_path = os.path.join('predictions/',filename)
        cv2.imwrite(predicted_img_path,predicted_img)

        # Retornar el archivo de imagen como respuesta
        return send_file(predicted_img_path), 200
    # Si no se envió nada o la extensión de archivo no es aceptada
    # se retorna un mensaje de error con código 400
    return jsonify({'error': 'Formato de archivo no permitido'}), 400

@app.route("/api/ml/video", methods=['GET', 'POST'])
def prediccion_video():
     # Verificar si la solicitud POST tiene el archivo adjunto
    if 'vid_src' not in request.files:
        return jsonify({'error': 'No se encontró el archivo en la solicitud'}), 400

    file = request.files['vid_src']

    # Verificar si el archivo tiene un nombre y una extensión válidos
    if file.filename == '':
        return jsonify({'error': 'El archivo no tiene un nombre válido'}), 400

    if file and allowed_file(file.filename):
        # Guardar el archivo en la carpeta de carga
        filename = secure_filename(str(file.filename))
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        predicted_video_path = os.path.join('predictions/',filename)
        # Predecir el díametro
        resultado = predecir_diametro_video(WEIGHTS_PATH,file_path,TRAINED_IMG_SIZE, predicted_video_path)
        
        if resultado:
            return jsonify({
                'Correcto: ': 'El archivo de video '+str(filename)+' se ha procesado correctamente',
                'Directorio de almacenamiento: ': str(file_path)}), 200

        # Retornar el archivo de imagen como respuesta
        return jsonify({'Error: ', 'El archivo no se leyó correctamente'}), 400
    # Si no se envió nada o la extensión de archivo no es aceptada
    # se retorna un mensaje de error con código 400
    return jsonify({'error': 'Formato de archivo no permitido'}), 400

def predecir_diametro_img(weights_path:str, img_src_path:str, trained_img_size:int):
    model = load_model(weights_path)
    model.conf = MIN_CONFIDENCE

    # Carga de imagen desde el directorio donde se almacenó
    src_img = cv2.imread(img_src_path)
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

    # Predicción del resultado
    result = model(src_img,size=trained_img_size)
    
    # Conversión en dataframe pandas
    df=result.pandas()
    
    # Número de instancias identificadas en la imagen
    object_count = len(df.xyxy[0])

    # Extracción de las coordenadas de las cajas delimitadoras identificadas
    # Se extraen como enteros para dibujarlas con OpenCV(cv2)
    df = df.xyxy[0][["xmin","ymin","xmax","ymax"]].values.astype(int)

    # Bucle para calculo de anchura, altura, coordenadas del centro de la caja 
    # y radio del circulo inscrito en la caja
    for i in range(object_count):
        x1, y1, x2, y2 = df[i][0], df[i][1], df[i][2], df[i][3]
        # Caja delimitadora
        #cv2.rectangle(src_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Anchura
        width = x2 - x1
        # Altura
        height = y2 - y1
        # Coordenadas x e y del centro de la caja
        center_x = int(x1 + (width/2))
        center_y = int(y1 + (height/2))
        # Radio del círculo inscrito
        radius = max(width, height)/2
        # Dibujo del circulo inscrito
        cv2.circle(src_img,(center_x,center_y),int(radius),(255, 0, 0), 3)

        # Cálculo del diámetro mínimo y máximo del círculo inscrito 
        diam15 = 2*(radius / RATIO_PX_CM_15)
        diam25 = 2*(radius / RATIO_PX_CM_25)
        # Impresión por consola de los diámetros calculados
        print('\nDiam15: '+str(round(diam15,2)))
        print('\nDiam25: '+str(round(diam25,2)))

        # Etiquetas del diámetro minimo y máximo calculados dibujadas en la imagen resultante
        cv2.putText(src_img, "Diam(min): {} cm".format(round(diam15,2)), (x2 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 2)
        cv2.putText(src_img, "Diam(max): {} cm".format(round(diam25,2)), (x2 + 10, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 2)
    
    # Asignación de resultado a nueva variable e intercambio de canales de color
    # para respuesta legible
    res_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

    return res_img

def predecir_diametro_video(weights_path:str, vid_src_path:str, trained_img_size:int, predicted_video_store:str):
    model = load_model(weights_path)

    # Carga de imagen desde el directorio donde se almacenó
    # Abre el video original
    video = cv2.VideoCapture(vid_src_path)
    if not(video.isOpened()):
        print("Error el video no se ha logrado abrir")
        return False

    # Configura el codec y el objeto VideoWriter para guardar el video modificado
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(predicted_video_store, fourcc, 30, (int(video.get(3)), int(video.get(4))))

    # Lee el primer fotograma
    ret, frame = video.read()

    while ret:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Aquí puedes aplicar cualquier procesamiento que desees a cada fotograma
        results = model(img, size=trained_img_size)
        df=results.pandas()
        object_count = len(df.xyxy[0])
        df = df.xyxy[0][["xmin","ymin","xmax","ymax"]].values.astype(int)
        # Bucle para calculo de anchura, altura, coordenadas del centro de la caja 
        # y radio del circulo inscrito en la caja
        for i in range(object_count):
            x1, y1, x2, y2 = df[i][0], df[i][1], df[i][2], df[i][3]
            # Anchura
            width = x2 - x1
            # Altura
            height = y2 - y1
            # Coordenadas x e y del centro de la caja
            center_x = int(x1 + (width/2))
            center_y = int(y1 + (height/2))
            # Radio del círculo inscrito
            radius = max(width, height)/2
            # Dibujo del circulo inscrito
            cv2.circle(img,(center_x,center_y),int(radius),(255, 0, 0), 3)

            # Cálculo del diámetro mínimo y máximo del círculo inscrito 
            diam15 = 2*(radius / RATIO_PX_CM_15)
            diam25 = 2*(radius / RATIO_PX_CM_25)

            # Etiquetas del diámetro minimo y máximo calculados dibujadas en la imagen resultante
            cv2.putText(img, "Diam(min): {} cm".format(round(diam15,2)), (x2 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 2)
            cv2.putText(img, "Diam(max): {} cm".format(round(diam25,2)), (x2 + 10, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 2)
        # Escribe el fotograma modificado en el objeto VideoWriter
        res_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out.write(res_img)

        # Lee el siguiente fotograma
        ret, frame = video.read()

    # Libera los objetos de video
    video.release()
    out.release()

    return True

def load_model(weights_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, _verbose=False, force_reload=True)
    return model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == "__main__":
    app.run()