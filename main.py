import cv2
import os
import image_processing as imProc  # Asegúrate de que el archivo image_processing.py esté en el mismo directorio o en el path de Python

# Función para leer una imagen y convertirla a escala de grises
def read_and_process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"No se puede cargar la imagen en la ruta: {image_path}")
    
    # Normalizar la imagen a rango [0, 1]
    return image.astype(float) / 255.0


# Función para guardar la imagen en el directorio correspondiente a la función de procesamiento
def save_processed_image(image, output_base_dir, filename, processing_function):
    func_name = processing_function.__name__  # Nombre de la función de procesamiento
    output_directory = os.path.join(output_base_dir, func_name)  # Crear subdirectorio para cada función
    os.makedirs(output_directory, exist_ok=True)
    
    output_path = os.path.join(output_directory, filename)  # Ruta de salida
    image_to_save = (image * 255).astype('uint8')  # Convertir de vuelta a uint8 para guardar
    cv2.imwrite(output_path, image_to_save)


# Rutas de los directorios de entrada y salida
input_directory = "/home/plegide/Documents/FIC/4/VA/in_pruebas"
output_base_directory = "/home/plegide/Documents/FIC/4/VA/out_pruebas"

# Diccionario que asocia funciones de procesamiento con sus parámetros específicos
processing_functions = {
    imProc.adjustIntensity: {"inRange": [], "outRange": [0.2, 0.8]},
    imProc.equalizeIntensity: {"nBins": 128},
    imProc.filterImage: {"kernel": [[1/9, 1/9, 1/9],
                                     [1/9, 1/9, 1/9],
                                     [1/9, 1/9, 1/9]]},
    imProc.gaussianFilter: {"sigma": 1.0},  # Cambia el valor de sigma según sea necesario
    imProc.medianFilter: {"filterSize": 3}  # Cambia el valor de filterSize según sea necesario
}

# Procesar todas las imágenes en el directorio de entrada
for filename in os.listdir(input_directory):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Filtrar por formatos de imagen
        input_image_path = os.path.join(input_directory, filename)

        # Leer y procesar la imagen
        image = read_and_process_image(input_image_path)

        # Aplicar funciones de procesamiento con sus respectivos parámetros
        for processing_function, params in processing_functions.items():
            # Aplicar la función de procesamiento con los parámetros específicos
            processed_image = processing_function(image, **params)

            # Guardar la imagen procesada en la carpeta correspondiente
            save_processed_image(processed_image, output_base_directory, filename, processing_function)

print("El procesamiento de imágenes ha finalizado.")
