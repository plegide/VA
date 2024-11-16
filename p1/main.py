import cv2
import os
import image_processing as imProc
import numpy as np


def read_and_process_image(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Leer la imagen en escala de grises
    if image is None:
        raise ValueError(f"No se puede cargar la imagen en la ruta: {image_path}")

    return image / 255.0 # Normalizar la imagen a [0, 1]


def save_processed_image(image, output_base_dir, filename, processing_function):

    func_name = processing_function.__name__
    output_directory = os.path.join(output_base_dir, func_name) # Se crea una carpeta para cada función de procesamiento
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, filename)
    image_to_save = image * 255.0  # Escalar la imagen a [0, 255] y convertirla a uint8
    cv2.imwrite(output_path, image_to_save)

def save_processed_tuple(image, output_base_dir, filename, processing_function, axis):

    func_name = processing_function.__name__ + axis
    output_directory = os.path.join(output_base_dir, func_name) # Se crea una carpeta para cada función de procesamiento
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, filename)
    image_to_save = image * 255.0  # Escalar la imagen a [0, 255] y convertirla a uint8
    #np.clip(image_to_save, 0, 255)
    cv2.imwrite(output_path, image_to_save)


# Rutas de los directorios de entrada y salida
# input_directory = "/home/plegide/Documents/FIC/4/VA/p1/in_pruebas"
# output_base_directory = "/home/plegide/Documents/FIC/4/VA/p1/out_pruebas"

input_directory = "/home/plegide/Documents/FIC/4/VA/p1/small_sample"
output_base_directory = "/home/plegide/Documents/FIC/4/VA/p1/small_result"

processing_functions = { # Diccionario de parametros para cada funcion
    #imProc.adjustIntensity: {"inRange": [], "outRange": [0.2, 0.8]},
    #imProc.equalizeIntensity: {"nBins": 128},
    # imProc.filterImage: {"kernel": [[1/9, 1/9, 1/9],
    #                                  [1/9, 1/9, 1/9],
    #                                  [1/9, 1/9, 1/9]]},
    #imProc.gaussianFilter: {"sigma": 0.8},
    #imProc.medianFilter: {"filterSize": 7}
    #  imProc.erode: {
    #      "SE": np.array([[0, 0, 1, 0, 0],
    #                     [0, 1, 1, 1, 0],
    #                     [1, 1, 1, 1, 1],
    #                     [0, 1, 1, 1, 0],
    #                     [0, 0, 1, 0, 0]]),
    #      "center": []
    #  },
    #  imProc.dilate: {
    #      "SE": np.array([[0, 0, 0],
    #                     [1, 1, 1],
    #                     [0, 0, 0]]),  # Elemento estructurante 3x3
    #      "center": []  # Calcula el centro automáticamente
    #  },
    # imProc.opening: {
    #     "SE": np.array([[1, 1, 1],
    #                     [1, 1, 1],
    #                     [1, 1, 1]]),
    #     "center": []  # Calcula el centro automáticamente
    # },
    # imProc.closing: {
    #     "SE": np.array([[1, 1, 1],
    #                     [1, 1, 1],
    #                     [1, 1, 1]]),
    #     "center": []  # Calcula el centro automáticamente
    # },
    # imProc.fill: {
    #     "seeds": np.array([[7,3]]),
    #     "SE": np.array([[1, 1, 1]]),
    #     "center": []
    # },
    #  imProc.gradientImage: {
    #      "operator": "Sobel"  # Puede ser Sobel, Roberts CentralDiff o Prewitt
    #  },
    # imProc.LoG: {
    #     "sigma": 1.0
    # },
    imProc.edgeCanny: {
        "sigma": 0.8,
        "tlow": 0.1,
        "thigh": 0.3
    }
}

for filename in os.listdir(input_directory):  # Todos los archivos en el directorio de entrada
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Solo se procesan archivos de imagen
        input_image_path = os.path.join(input_directory, filename)
        image = read_and_process_image(input_image_path)
        
        for processing_function, params in processing_functions.items():
            if processing_function.__name__ == "gradientImage":
                processed_imageX, processed_imageY = processing_function(image, **params) # Aplicar funciones y parámetros del diccionario 
                save_processed_tuple(processed_imageX, output_base_directory, filename , processing_function, "X")
                save_processed_tuple(processed_imageY, output_base_directory, filename, processing_function, "Y")
            else:
                processed_image = processing_function(image, **params)
                #processed_image = imProc.adjustIntensity(processed_image, [], [0, 1])
                save_processed_image(processed_image, output_base_directory, filename, processing_function)

print("FIN")


#img = cv2.imread("/home/plegide/Documents/FIC/4/VA/p1/in_pruebas/image2.png", cv2.IMREAD_GRAYSCALE)
#imgCV = cv2.filter2D(img, -1, np.array(imProc.gaussKernel1D(0.8)))
#imgCV = cv2.filter2D(imgCV, -1, np.array(imProc.gaussKernel1D(0.8)).T)
#cv2.imwrite("/home/plegide/Documents/FIC/4/VA/p1/out_pruebas/cv2.png", imgCV)


# def create_image_from_list(pixel_data, output_path):
#     """
#     Crea y guarda una imagen en escala de grises a partir de una lista de listas binaria.
    
#     Args:
#         pixel_data (list of list of int): Lista de listas que contiene 1 para píxeles blancos y 0 para píxeles negros.
#         output_path (str): Ruta para guardar la imagen resultante.
#     """
#     # Convertir la lista de listas en un array de numpy
#     pixel_array = np.array(pixel_data, dtype=np.uint8) * 255  # Multiplicamos por 255 para que 1 sea blanco (255) y 0 sea negro (0)
    
#     # Guardar la imagen
#     cv2.imwrite(output_path, pixel_array)

# Probar con SE cuadrado, circulo y cruz
# Ejemplo dilate con 0s

# pixel_data = [
#     [1, 0, 0, 0],
#     [1, 0, 0, 0],
#     [0, 1, 1, 0],
#     [0, 1, 0, 0],
#     [0, 1, 0, 0],
# ]

# Ejemplo fill diapositivas

# pixel_data = [  [0, 0, 0, 1, 1, 0, 0, 0],
#                 [0, 0, 0, 1, 0, 1, 0, 0],
#                 [0, 0, 1, 0, 0, 1, 0, 0],
#                 [0, 0, 1, 0, 0, 1, 0, 0],
#                 [0, 1, 0, 0, 0, 1, 0, 0],
#                 [0, 1, 0, 0, 0, 1, 0, 0],
#                 [0, 0, 1, 0, 1, 0, 0, 0],
#                 [0, 0, 0, 1, 0, 0, 0, 0]
#             ]


# output_path = "/home/plegide/Documents/FIC/4/VA/p1/in_pruebas/morphGenerada.png"
# create_image_from_list(pixel_data, output_path)
# image = read_and_process_image(output_path)
# morphed_image = imProc.dilate(image, np.array([[1,0,1]]), [])
# morphed_image = imProc.fill(image,[(4,3)],np.array([[0,1,0], [1,1,1], [0,1,0]]))
# image_to_save = (morphed_image * 255).astype('uint8')
# cv2.imwrite("/home/plegide/Documents/FIC/4/VA/p1/out_pruebas/morfDilated.png", image_to_save)
# cv2.imwrite("/home/plegide/Documents/FIC/4/VA/p1/out_pruebas/morfFilled.png", image_to_save)
# print("FIN")