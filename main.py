import cv2
import os
from image_processing import adjustIntensity, equalizeIntensity

# Función para leer una imagen y convertirla a escala de grises
def read_and_process_image(image_path):
    # Leer la imagen en escala de grises
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"No se puede cargar la imagen en la ruta: {image_path}")
    
    # Normalizar la imagen a rango [0, 1]
    return image.astype(float) / 255.0


# Función para guardar la imagen en el directorio correspondiente a la función de procesamiento
def save_processed_image(image, output_base_dir, filename, processing_function):
    # Obtener el nombre de la función de procesamiento
    func_name = processing_function.__name__
    
    # Crear un subdirectorio basado en el nombre de la función de procesamiento
    output_directory = os.path.join(output_base_dir, func_name)
    os.makedirs(output_directory, exist_ok=True)
    
    # Guardar la imagen en el directorio correspondiente
    output_path = os.path.join(output_directory, filename)
    image_to_save = (image * 255).astype('uint8')
    cv2.imwrite(output_path, image_to_save)


# Rutas de los directorios de entrada y salida
input_directory = "/home/plegide/Documents/FIC/4/VA/in_pruebas"
output_base_directory = "/home/plegide/Documents/FIC/4/VA/out_pruebas"

# Diccionario que asocia funciones de procesamiento con sus parámetros específicos
processing_functions = {
    adjustIntensity: {"inRange": [], "outRange": [0.2, 0.8]},
    equalizeIntensity: {"nBins": 128}  # Cambié el valor predeterminado a 128 para ilustrar
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
