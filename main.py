import cv2
import os
import image_processing as imProc
import numpy as np


def read_and_process_image(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Leer la imagen en escala de grises
    if image is None:
        raise ValueError(f"No se puede cargar la imagen en la ruta: {image_path}")

    return image.astype(float) / 255.0 # Normalizar la imagen a [0, 1]


def save_processed_image(image, output_base_dir, filename, processing_function):

    func_name = processing_function.__name__
    output_directory = os.path.join(output_base_dir, func_name) # Se crea una carpeta para cada función de procesamiento
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, filename) 
    image_to_save = (image * 255).astype('uint8')  # Escalar la imagen a [0, 255] y convertirla a uint8
    cv2.imwrite(output_path, image_to_save)


# Rutas de los directorios de entrada y salida
input_directory = "/home/plegide/Documents/FIC/4/VA/in_pruebas"
output_base_directory = "/home/plegide/Documents/FIC/4/VA/out_pruebas"


processing_functions = { # Diccionario de parametros para cada funcion
    #imProc.adjustIntensity: {"inRange": [], "outRange": [0.2, 0.8]},
    #imProc.equalizeIntensity: {"nBins": 128},
    #imProc.filterImage: {"kernel": [[1/9, 1/9, 1/9],
    #                                 [1/9, 1/9, 1/9],
    #                                 [1/9, 1/9, 1/9]]},
    imProc.gaussianFilter: {"sigma": 0.8},
    #imProc.medianFilter: {"filterSize": 7}  
}


for filename in os.listdir(input_directory): # Todos los archivos en el directorio de entrada
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Solo se procesan archivos de imagen
        input_image_path = os.path.join(input_directory, filename)
        image = read_and_process_image(input_image_path)
        for processing_function, params in processing_functions.items(): # Aplicar funciones y parametros del diccionario
            
            processed_image = processing_function(image, **params)
            save_processed_image(processed_image, output_base_directory, filename, processing_function)

print("FIN")

img = cv2.imread("/home/plegide/Documents/FIC/4/VA/in_pruebas/image2.png", cv2.IMREAD_GRAYSCALE)
imgCV = cv2.filter2D(img, -1, np.array(imProc.gaussKernel1D(0.8)))
imgCV = cv2.filter2D(imgCV, -1, np.array(imProc.gaussKernel1D(0.8)).T)
cv2.imwrite("/home/plegide/Documents/FIC/4/VA/out_pruebas/cv2.png", imgCV)

