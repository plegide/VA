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