import tensorflow as tf
import numpy as np
import keras.saving  
import json

import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tqdm import tqdm

from labels_transform import create_transformer, code_word, decode_word

from list_pictures import list_pictures
from diagnostic_load import full_data_diagnosis_load, load_diagnostic

from visualize_image import visualize_image


DATA_DIR = "classification_proyect/code/images"
DATA_FILE = "classification_proyect/code/BBox_List_2017.csv"

visualizeImage = False



def load_data(data_dir:str='', target_size:tuple=(224, 224)):

    #declaracion de los arreglos vacios
    images = []
    features = []
    labels = []

    full_data = full_data_diagnosis_load(path_csv=DATA_FILE)

    for file_name in tqdm(list_pictures(path=DATA_DIR)):
        img_path = os.path.join(DATA_DIR, file_name) # concatena el path para poder abrir la imagen
        try:
            disease, x, y, w, h  = load_diagnostic(file_name=file_name, full_data=full_data) #extrae la informacion por coincidencia del nombre de la imagen

            if visualizeImage:
                visualize_image(img_path, x=x, y=y, width=w, height=h)

            img = load_img(img_path, target_size=target_size, color_mode='grayscale')#carga la imagen redimencionando y convirtiendo a escala de grises
            img_array = img_to_array(img) #convierte la imagen jpg en un arreglo , matriz tridimencional, red[], green[], blue[] si es rgb, solo una matriz si es en escala de grises
            # img = img.reshape((1,) + img.shape)
            img_normalized = img_array.astype(np.float32) / 255.0 ,  #255 normaliza entre 0 y 1 para cada matriz


            images.append(img_normalized[0])
            labels.append(disease)
            features.append([float(x), float(y), float(w), float(h)])
        except Exception as e:
            pass

    return (np.array(images), np.array(features), np.array(labels))


if __name__ == '__main__':
    # breakpoint()
    #cargar los datos con caracteristicas
    img_size = (512, 512) #224
    X_images, X_features, y_labels = load_data(data_dir=DATA_DIR, target_size=img_size)

    breakpoint()
    diseases = list(set(y_labels))
    n_labels = len(diseases)

    label_transformer = create_transformer(list_words=diseases)

    new_y_labels = []
    for label in y_labels:
        codeword = code_word(word=label, encoder=label_transformer)
        new_y_labels.append(codeword)

    y_labels = new_y_labels

    print("*"*50)
    print(f"cantidad de etiquetas : {n_labels}")
    print(f"categorias encontradas: {diseases}")
    print(f"cantidad de archivos : {len(X_images)}")
    print("*"*50)

    # Normalizar características y convertir etiquetas a categóricas

    # for feature in X_features:
    #     print(f"featre : {feature} type : {type(feature)}")
    #     print(f"type[0] :{type(feature[0])}")
    #     print(f"type[1] :{type(feature[0])}")
    #     print(f"type[2] :{type(feature[0])}")
    #     print(f"type[3] :{type(feature[0])}")
    # X_features = X_features / np.array([img_size[0], img_size[1], img_size[0], img_size[1]])
    y_labels = to_categorical(y_labels, num_classes=n_labels)

    # Definir modelo
    image_input = tf.keras.layers.Input(shape=(img_size[0], img_size[1], 1)) #configurada para imagenes en escala de grises
    #image_input = tf.keras.layers.Input(shape=(img_size[0], img_size[1], 3)) #configurada para imagenes rgb
    feature_input = tf.keras.layers.Input(shape=(4,)) #define la seguda entrada un vector de 4 caracteristicas por muestra

    #cnn : "Convolutional Neural Network" declaracion del modelo
    # cnn = Sequential([
    #     Conv2D(32, (3, 3), activation='relu', input_shape=( img_size[0], img_size[1], 3)), #capa convolucional con 32 filtros de tamaño 3x3
    #     MaxPooling2D((2, 2)),# reduce el tamaño de la imagen a la mitad (redimenciona el areglo 2D)
    #     Conv2D(64, (3, 3), activation='relu'), # segunga capa convolucional de 64 filtros de tamaño 3x3
    #     MaxPooling2D((2, 2)), # reduce el tamaño de la imagen a la mitad nuevamente (redimenciona el areglo 2D)
    #     Flatten() #convierte la salida convolucionada en un vector unidimensional para que pueda conectarse con capas densas totalmente conectadas(FC)
    # ])

    cnn = Sequential([
    # Capa convolucional con 32 filtros de tamaño 3x3, entrada de una imagen en escala de grises de tamaño (1024, 1024, 1)
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)),
    # Capa de MaxPooling que reduce el tamaño de la imagen a la mitad (2x2)
    MaxPooling2D((2, 2)),
    # Segunda capa convolucional con 64 filtros de tamaño 3x3
    Conv2D(64, (3, 3), activation='relu'),
    # Segunda capa de MaxPooling que reduce el tamaño de la imagen nuevamente a la mitad
    MaxPooling2D((2, 2)),
    # Aplana la salida convolucionada en un vector unidimensional para que pueda conectarse a capas densas
    Flatten()
    ])


    cnn_output = cnn(image_input) #aplicar la redneuronal al array de imagenes
    merged = tf.keras.layers.concatenate([cnn_output, feature_input]) #fusiona las imagenes con caracteristica adicionales

    output = Dense(128, activation='relu')(merged) #Primera capa totalmente conectada con 128 neuronas y activación ReLU.
    output = Dropout(0.5)(output) #Apaga el 50% de las neuronas en cada iteración para evitar sobreajuste.
    output = Dense(n_labels, activation='softmax')(output) #Capa de salida con 7 categorias

    model = tf.keras.Model(inputs=[image_input, feature_input], outputs=output) #Se crea un modelo con dos entradas (image_input y feature_input) y una salida (output).

    # Compilar modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # optimizer='adam': Usa el optimizador Adam, que es eficiente para redes profundas.
    # loss='categorical_crossentropy': Pérdida usada en clasificación multiclase con etiquetas one-hot.
    # metrics=['accuracy']: Mide la precisión del modelo.

    # Resumen del modelo
    # model.summary()

    # print("tipo de imagenes", X_images.dtype)  # Debería ser algo como float32 o float64
    # print("tipo de caracteristicas", X_features.dtype)  # Igualmente, debería ser numérico
    # print("tipo de etiquetas", y_labels.dtype)

    # Entrenar modelo
    history = model.fit([X_images, X_features], y_labels, epochs=50, batch_size=50, validation_split=0.2)
    # X_images: Datos de imágenes procesadas (224x224x3).
    # X_features: Datos adicionales (vector de 4 valores).
    # y_labels: Etiquetas en formato one-hot con 15 categorías.
    # epochs=10: Se entrena durante 10 iteraciones completas sobre los datos.
    # batch_size=32: Entrena con lotes de 32 muestras a la vez.
    # validation_split=0.2: Usa 20% de los datos para validación.


    # Guardar el history en un archivo JSON
    with open("training_history.json", "w") as f:
        json.dump(history.history, f)


    # Guardar el modelo
    keras.saving.save_model(model, "lung_prediction_model.keras")


