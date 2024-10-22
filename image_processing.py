import numpy as np
import matplotlib.pyplot as plt


def adjustIntensity(inImage, inRange=[], outRange=[0, 1]):

    if len(inRange) == 0: # Si no se especifica el rango de entrada, se saca de la imagen
        imin, imax = np.min(inImage), np.max(inImage)
    else:
        imin, imax = inRange
    omin, omax = outRange # Dividir el rango de salida en los valores mínimo y máximo

    outImage = (inImage - imin) / (imax - imin) * (omax - omin) + omin # Aplicar la transformación lineal
    
    return outImage


def equalizeIntensity(inImage, nBins=256):

    height, width = inImage.shape

    hist = np.zeros(nBins)
    for i in range(height): # Calcular el histograma de la imagen
        for j in range(width):
            pixel_value = int(inImage[i, j] * (nBins - 1))  # Normalizar el valor del pixel a [0, nBins - 1]
            hist[pixel_value] += 1

    cdf = hist.cumsum() # Función de distribución acumulada
    cdf_normalized = cdf / cdf[-1] # Normalizar la CDF
    outImage = np.interp(inImage.flatten(), np.linspace(0, 1, nBins), cdf_normalized) # Interpolar la CDF para obtener la imagen de salida
    outImage = outImage.reshape(inImage.shape) # Reajustar las dimensiones de la imagen

    return outImage


def filterImage(inImage, kernel):

    kernel = np.array(kernel)  # Convertir el kernel a un array de numpy
    img_height, img_width = inImage.shape
    k_height, k_width = kernel.shape
    pad_height = k_height // 2 # Calcular el padding
    pad_width = k_width // 2

    outImage = np.zeros_like(inImage) 
    padded_image = np.pad(inImage, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect') # Aplicar padding
    for i in range(img_height): # Convolucion
        for j in range(img_width):
            region = padded_image[i:i + k_height, j:j + k_width] # Extraer la región correspondiente de la imagen
            outImage[i, j] = np.sum(region * kernel) # Calcular la convolución y asignarla a la imagen de salida

    return outImage



def gaussKernel1D(sigma):
    N = int(2 * np.ceil(3 * sigma) + 1)  # Calcular N a partir de sigma
    x = np.linspace(-(N // 2), N // 2, N)  # Crear un vector de índice centrado
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))  # Calcular el kernel gaussiano
    kernel /= np.sum(kernel)  # Normalizar el kernel

    # Plotear la campana de gauss
    plt.figure(figsize=(8, 4))
    plt.plot(x, kernel, label=f'Sigma = {sigma}', color='blue')
    plt.title('Campana de Gauss')
    plt.xlabel('x')
    plt.ylabel('Valor del Kernel')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.grid()
    plt.legend()
    plt.savefig(f'kernel_gaussiano.png')
    plt.close()

    return kernel

def gaussianFilter(inImage, sigma):

    kernel_1d = gaussKernel1D(sigma)
    temp_image = filterImage(inImage, kernel_1d.reshape(1, -1))  # Aplicar como filtro 1xN
    outImage = filterImage(temp_image, kernel_1d.reshape(-1, 1))  # Aplicar como filtro Nx1 traspuesto

    return outImage

def medianFilter(inImage, filterSize):

    img_height, img_width = inImage.shape
    pad_size = filterSize // 2 # Calcular padding
    outImage = np.zeros_like(inImage)
    padded_image = np.pad(inImage, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')

    for i in range(img_height): # Calcular la mediana
        for j in range(img_width):
            region = padded_image[i:i + filterSize, j:j + filterSize] # Extraer la región de la imagen
            outImage[i, j] = np.median(region) # Calcular la mediana de la region y asignarla a la imagen de salida
    
    return outImage
