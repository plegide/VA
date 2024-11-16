import numpy as np
from collections import deque
import matplotlib.pyplot as plt


# ----------------------------------------- HISTOGRAMAS: MEJORA DE CONTRASTE -----------------------------------------


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


# ----------------------------------------- FILTRADO ESPACIAL: SUAVIZADO -----------------------------------------


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


def plotGaussKernel(x, kernel, sigma):
    """
    Genera el gráfico de una campana de Gauss a partir de un kernel.

    Parameters:
        x (np.ndarray): Vector de índices centrado.
        kernel (np.ndarray): Kernel gaussiano 1D.
        sigma (float): Desviación estándar de la distribución gaussiana.
    """
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

def gaussKernel1D(sigma):

    N = int(2 * np.ceil(3 * sigma) + 1)  # Calcular N a partir de sigma
    x = np.linspace(-(N // 2), N // 2, N)  # Crear un vector de índice centrado
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))  # Calcular el kernel gaussiano
    kernel /= np.sum(kernel)  # Normalizar el kernel

    # Plotear la campana de gauss
    plotGaussKernel(x, kernel, sigma)

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


# ----------------------------------------- OPERADORES MORFOLOGICOS -----------------------------------------


def erode(inImage, SE, center=[]):

    img_height, img_width = inImage.shape
    se_height, se_width = SE.shape
    if not center:
        center = [(se_height // 2), (se_width // 2)] 
    center_x, center_y = center
    
    pad_top = center_x  # El padding se calcula en funcion del centro del SE
    pad_bottom = se_height - center_x - 1
    pad_left = center_y
    pad_right = se_width - center_y - 1
    padded_image = np.pad(inImage, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant') # Padding con 0s
    outImage = np.zeros_like(inImage)

    for i in range(img_height):
        for j in range(img_width):
            if inImage[i, j] == 1:  # Buscar los 1s en la imagen de entrada
                region = padded_image[i:i + se_height, j:j + se_width] # Mismo tamaño que el SE
                if np.all(region[SE == 1] == 1): # Coges los 1 del SE y miras si en la posicion en la que estan en el trozo del a original es 1
                    outImage[i, j] = 1
    
    return outImage


def dilate(inImage, SE, center=[]):

    img_height, img_width = inImage.shape
    se_height, se_width = SE.shape
    if not center:
        center = [(se_height // 2), (se_width // 2)]
    center_x, center_y = center
    pad_top = center_x # Calcular el padding en función del centro del SE
    pad_bottom = se_height - center_x - 1
    pad_left = center_y
    pad_right = se_width - center_y - 1
    padded_image = np.pad(inImage, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    dilated_image = np.zeros_like(padded_image, dtype=np.uint8)

    for i in range(img_height):
        for j in range(img_width):
            if inImage[i, j] == 1:  # En los 1s de la imagen original
                dilated_image[i:i + se_height, j:j + se_width] = np.maximum(dilated_image[i:i + se_height, j:j + se_width], SE) # Pones los 1s del SE sin sobreescribir

    outImage = dilated_image[pad_top:pad_top + img_height, pad_left:pad_left + img_width] # El padding no se tiene en cuenta
    return outImage


def opening(inImage, SE, center=[]):

    return dilate(erode(inImage, SE, center), SE, center) 


def closing(inImage, SE, center=[]):

    return erode(dilate(inImage, SE, center), SE, center)


def fill(inImage, seeds, SE=[], center=[]):
    """
    Rellena una región basada en el algoritmo descrito usando dilatación condicional.

    Args:
        inImage: Matriz binaria (0 y 1) que representa la imagen de entrada.
        seeds: Matriz Nx2 con N coordenadas (fila, columna) de los puntos semilla.
        SE: Matriz PxQ de 0s y 1s definiendo el elemento estructurante de conectividad.
            Si es una lista vacía, se asume conectividad-4 (cruz 3x3).
        center: Coordenadas (fila, columna) del centro del elemento estructurante.
                Si es una lista vacía, se calcula automáticamente como el centro geométrico de SE.

    Returns:
        outImage: Imagen binaria con la región rellenada.
    """

    img_height, img_width = inImage.shape

    # Usar conectividad-4 si no se proporciona un elemento estructurante
    if isinstance(SE, list) and len(SE) == 0:  # Si SE es una lista vacía
        SE = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=bool)

    se_height, se_width = SE.shape

    # Calcular el centro del elemento estructurante si no se proporciona
    if not center:
        center = (se_height // 2, se_width // 2)

    center_x, center_y = center

    # Calcular manualmente el complemento de la imagen de entrada
    A_complement = np.zeros_like(inImage, dtype=int)
    for i in range(img_height):
        for j in range(img_width):
            A_complement[i, j] = 1 if inImage[i, j] == 0 else 0

    # Crear una imagen binaria para el resultado
    outImage = np.zeros_like(inImage, dtype=bool)

    # Inicializar los puntos semilla en la imagen de salida
    for seed in seeds:
        outImage[seed[0], seed[1]] = 1

    # Realizar el llenado de regiones
    while True:
        # Dilatación del conjunto actual (X_k-1) con el elemento estructurante
        dilated = np.zeros_like(outImage, dtype=bool)
        for i in range(img_height):
            for j in range(img_width):
                if outImage[i, j]:  # Si el píxel pertenece a X_k-1
                    # Aplicar el elemento estructurante
                    for di, row in enumerate(SE):
                        for dj, val in enumerate(row):
                            if val:  # Considerar solo los puntos estructurantes no nulos
                                ni, nj = i + di - center_x, j + dj - center_y
                                if 0 <= ni < img_height and 0 <= nj < img_width:
                                    dilated[ni, nj] = True

        # Intersección con el complemento de la imagen de entrada
        nextImage = np.zeros_like(dilated, dtype=int)
        for i in range(img_height):
            for j in range(img_width):
                nextImage[i, j] = 1 if dilated[i, j] == 1 and A_complement[i, j] == 1 else 0

        # Terminar si no hay cambios entre iteraciones
        if np.array_equal(nextImage, outImage):
            break

        # Actualizar la imagen de salida
        outImage = nextImage

    # Unión final con la imagen original
    for i in range(img_height):
        for j in range(img_width):
            outImage[i, j] = 1 if outImage[i, j] == 1 or inImage[i, j] == 1 else 0

    return outImage


# ----------------------------------------- DETECCION DE BORDES  -----------------------------------------


def gradientImage(inImage, operator):

    if operator == 'Roberts':
        Gx = np.array([[1, 0], [0, -1]])
        Gy = np.array([[0, 1], [-1, 0]])

    elif operator == 'CentralDiff':
        Gx = np.array([[-1, 0, 1]])
        Gy = np.array([[-1], [0], [1]])

    elif operator == 'Prewitt':
        Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        Gy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    elif operator == 'Sobel':
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    else:
        raise ValueError("El kernel tiene que ser: 'Roberts', 'CentralDiff', 'Prewitt' o 'Sobel'")

    gx = filterImage(inImage, Gx)
    gy = filterImage(inImage, Gy)
    return [gx, gy]


def LoG(inImage, sigma):
    # Calcular tamaño del kernel
    N = int(2 * np.ceil(3 * sigma) + 1)  # Tamaño del kernel basado en sigma
    kernel = np.zeros((N, N))  # Inicializar el kernel de tamaño N x N

    
    # Coordenadas del centro del kernel
    center = N // 2

    # Llenar el kernel LoG
    for x in range(N):
        for y in range(N):
            # Calcular x_rel y y_rel como distancia al centro del kernel
            x_rel = x - center
            y_rel = y - center
            
            # Calcular valor del Laplaciano de Gaussiano
            kernel[x, y] = ((x_rel**2 + y_rel**2 - sigma**2) / sigma**4) * np.exp(-(x_rel**2 + y_rel**2) / (2 * sigma**2))

    # Aplicar el filtro a la imagen de entrada usando convolución
    kernel = kernel - np.mean(kernel)  # Restar la media del kernel
    outImage = filterImage(inImage, kernel)
    
    return outImage


def nonMaximumSuppression(magnitude, gradient_direction):

    img_height, img_width = magnitude.shape
    suppressed_image = np.zeros((img_height, img_width), dtype=np.float32)
    gradient_direction = gradient_direction % 180 # Normalizar el angulo a [0,180)
    
    for i in range(1, img_height - 1):
        for j in range(1, img_width - 1):
            angle = gradient_direction[i, j]
            current_magnitude = magnitude[i, j]   
            # Comparar cada pixel con el que esta en la direccion del gradiente
            if (0 <= angle < 22.5) or (157.5 <= angle < 180): # 0 o 180 vecinos en los lados izquierdo y derecho
                neighbor_1 = magnitude[i, j - 1]
                neighbor_2 = magnitude[i, j + 1]
            elif 22.5 <= angle < 67.5: # 45 vecinos en las posiciones diagonal superior izquierda e inferior derecha
                neighbor_1 = magnitude[i - 1, j - 1]
                neighbor_2 = magnitude[i + 1, j + 1]
                
            elif 67.5 <= angle < 112.5: # 90 vecinos arriba y abajo
                neighbor_1 = magnitude[i - 1, j]
                neighbor_2 = magnitude[i + 1, j]
            else:  # 112.5 <= angle < 157.5 135  vecinos en las posiciones diagonal superior derecha e inferior izquierda
                neighbor_1 = magnitude[i - 1, j + 1]
                neighbor_2 = magnitude[i + 1, j - 1]

            if current_magnitude >= neighbor_1 and current_magnitude >= neighbor_2: # Si no es maximo
                suppressed_image[i, j] = current_magnitude

    return suppressed_image


def hysteresis(suppressed_image, gradient_direction, tlow, thigh, neighbor_depth=3):
    output_image = np.zeros_like(suppressed_image)
    img_height, img_width = suppressed_image.shape
    gradient_direction = gradient_direction % 180  # Normalizar el angulo a [0,180)

    queue = deque()
    for i in range(1, img_height - 1):
        for j in range(1, img_width - 1):
            if suppressed_image[i, j] >= thigh: # Bordes fuertes
                output_image[i, j] = 1.0
                queue.append((i, j))

    while queue: # Mientras haya pixeles que procesar
        i, j = queue.popleft()
        angle = gradient_direction[i, j]
        neighbors = []
        # Buscar 3 vecinos en la direccion perpendicular a la normal, la direccion del borde, por robustez
        if (0 <= angle < 22.5) or (157.5 <= angle < 180):  # Arriba y abajo
            for offset in range(1, neighbor_depth + 1):
                neighbors.extend([(i - offset, j), (i + offset, j)])
        elif 22.5 <= angle < 67.5:  # Diagonales superior derecha e inferior izquierda
            for offset in range(1, neighbor_depth + 1):
                neighbors.extend([(i - offset, j + offset), (i + offset, j - offset)])
        elif 67.5 <= angle < 112.5:  # Izquierda y derecha
            for offset in range(1, neighbor_depth + 1):
                neighbors.extend([(i, j - offset), (i, j + offset)])
        else:  # Diagonales superior izquierda e inferior derecha
            for offset in range(1, neighbor_depth + 1):
                neighbors.extend([(i + offset, j + offset), (i - offset, j - offset)])

        for ni, nj in neighbors: #Buscar bordes suaves en los vecinos
            if 0 <= ni < img_height and 0 <= nj < img_width:
                if suppressed_image[ni, nj] >= tlow and output_image[ni, nj] == 0: # Si no se ha marcado ya, marcar como borde fuerte y procesar
                    output_image[ni, nj] = 1.0
                    queue.append((ni, nj))

    return output_image


def edgeCanny(inImage, sigma, tlow, thigh):

    smoothed_image = gaussianFilter(inImage, sigma) #Suavizado gaussiano

    gx, gy = gradientImage(smoothed_image, "Sobel") # Gradiente con Sobel
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    gradient_direction = np.degrees(np.arctan2(gy, gx))

    suppressed_image = nonMaximumSuppression(magnitude, gradient_direction) # Supresión no máxima
    
    current_image = hysteresis(suppressed_image, gradient_direction, tlow, thigh)
    next_hysteresis = hysteresis(current_image, gradient_direction, tlow, thigh)

    while not np.array_equal(current_image, next_hysteresis):
        current_image = next_hysteresis
        next_hysteresis = hysteresis(current_image, gradient_direction, tlow, thigh)

    final_image = current_image

    
    # Visualizar los pasos
    plt.figure(figsize=(12, 10))
    plt.suptitle("Resultados de Canny")
    plt.subplot(2, 2, 1)
    plt.title("Suavizado Gaussiano")
    plt.imshow(smoothed_image, cmap="gray")
    plt.axis("off")
    plt.subplot(2, 2, 2)
    plt.title("Magnitud del Gradiente")
    plt.imshow(magnitude, cmap="gray")
    plt.axis("off")
    plt.subplot(2, 2, 3)
    plt.title("Supresión No Máxima")
    plt.imshow(suppressed_image, cmap="gray")
    plt.axis("off")
    plt.subplot(2, 2, 4)
    plt.title("Histeresis")
    plt.imshow(final_image, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return final_image