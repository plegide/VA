import numpy as np

# Función para validar que la imagen es una matriz 2D de punto flotante
def validate_image(image):
    if image.ndim != 2 or image.dtype not in [np.float32, np.float64]:
        raise ValueError("La imagen de entrada debe ser una matriz 2D de punto flotante.")


# Función para ajustar la intensidad (estiramiento lineal del histograma)
def adjustIntensity(inImage, inRange=[], outRange=[0, 1]):
    validate_image(inImage)  # Validar la imagen de entrada

    # Obtener los límites de entrada
    if len(inRange) == 0:
        imin, imax = np.min(inImage), np.max(inImage)
    else:
        imin, imax = inRange
    
    # Asegurarse de que los límites de salida estén bien definidos
    omin, omax = outRange
    
    # Realizar el ajuste lineal
    outImage = (inImage - imin) / (imax - imin) * (omax - omin) + omin
    
    # Clip para mantener los valores dentro del rango [0, 1]
    outImage = np.clip(outImage, omin, omax)
    
    return outImage


# Función para ecualizar el histograma
def equalizeIntensity(inImage, nBins=256):
    validate_image(inImage)  # Validar la imagen de entrada

    # Obtener las dimensiones de la imagen
    height, width = inImage.shape

    # Crear un histograma vacío
    hist = np.zeros(nBins)

    # Calcular el histograma manualmente
    for i in range(height):
        for j in range(width):
            pixel_value = int(inImage[i, j] * (nBins - 1))  # Escalar a [0, nBins-1]
            hist[pixel_value] += 1

    # Calcular la función de distribución acumulativa (CDF)
    cdf = hist.cumsum()

    # Normalizar la CDF
    cdf_normalized = cdf / cdf[-1]

    # Usar np.interp para mapear los valores de la imagen de entrada usando la CDF
    outImage = np.interp(inImage.flatten(), np.linspace(0, 1, nBins), cdf_normalized)

    # Volver a dar forma a la imagen de salida
    outImage = outImage.reshape(inImage.shape)

    return outImage


# Función para aplicar el filtrado espacial mediante convolución
def filterImage(inImage, kernel):
    kernel = np.array(kernel)  # Convertir el kernel a un array de numpy

    validate_image(inImage)  # Validar la imagen de entrada

    # Obtener las dimensiones de la imagen y del kernel
    img_height, img_width = inImage.shape
    k_height, k_width = kernel.shape

    # Calcular el margen del kernel
    pad_height = k_height // 2
    pad_width = k_width // 2

    # Crear una imagen de salida inicializada en ceros
    outImage = np.zeros_like(inImage)

    # Aplicar el padding a la imagen de entrada
    padded_image = np.pad(inImage, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Realizar la convolución
    for i in range(img_height):
        for j in range(img_width):
            # Extraer la región correspondiente de la imagen
            region = padded_image[i:i + k_height, j:j + k_width]
            # Aplicar la convolución (producto punto)
            outImage[i, j] = np.sum(region * kernel)

    return outImage


def gaussKernel1D(sigma):
    # Calcular N a partir de sigma
    N = int(2 * np.ceil(3 * sigma) + 1)
    
    # Crear un vector de índice centrado
    x = np.linspace(-(N // 2), N // 2, N)

    # Calcular el kernel gaussiano
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))

    # Normalizar el kernel
    kernel /= np.sum(kernel)
    
    return kernel

def gaussianFilter(inImage, sigma):
    validate_image(inImage)  # Validar la imagen de entrada

    # Crear el kernel gaussiano unidimensional
    kernel_1d = gaussKernel1D(sigma)

    # Convolucionar primero con el kernel unidimensional
    temp_image = filterImage(inImage, kernel_1d.reshape(1, -1))  # Aplicar como filtro 1xN

    # Convolucionar luego con el kernel transpuesto
    outImage = filterImage(temp_image, kernel_1d.reshape(-1, 1))  # Aplicar como filtro Nx1

    return outImage

def medianFilter(inImage, filterSize):
    validate_image(inImage)  # Validar la imagen de entrada

    # Obtener las dimensiones de la imagen
    img_height, img_width = inImage.shape

    # Calcular el margen del filtro
    pad_height = filterSize // 2
    pad_width = filterSize // 2

    # Crear una imagen de salida inicializada en ceros
    outImage = np.zeros_like(inImage)

    # Aplicar el padding a la imagen de entrada
    padded_image = np.pad(inImage, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')

    # Realizar el filtrado de mediana
    for i in range(img_height):
        for j in range(img_width):
            # Extraer la región correspondiente de la imagen
            region = padded_image[i:i + filterSize, j:j + filterSize]
            # Calcular la mediana y asignarla a la imagen de salida
            outImage[i, j] = np.median(region)
            
    return outImage