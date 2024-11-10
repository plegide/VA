import numpy as np
import matplotlib.pyplot as plt
import cv2


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
    img_height, img_width = inImage.shape
    
    # Usar conectividad 4 si no se proporciona un elemento estructurante
    if isinstance(SE, list) and len(SE) == 0:  # Verificar si SE es una lista vacía
        SE = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])
    
    se_height, se_width = SE.shape
    print(se_height, se_width)

    # Calcular el centro del SE si no se proporciona
    if not center:
        center = (se_height // 2, se_width // 2)
        print(center)
    
    center_x, center_y = center
    print(center_x, center_y)
    
    # Inicializar `outImage` como una copia de `inImage`
    outImage = inImage.copy()
    
    # Marcar los puntos semilla en `outImage`
    for seed in seeds:
        outImage[seed[0], seed[1]] = 1  # Marcar el punto semilla

    # Realizar el llenado
    changed = True
    while changed:
        changed = False
        for i in range(img_height):
            for j in range(img_width):
                if outImage[i, j] == 1:  # Solo considerar píxeles marcados
                    # Comprobar los vecinos según el SE, usando el centro
                    for di in range(-center_x, se_height - center_x):
                        for dj in range(-center_y, se_width - center_y):
                            ni, nj = i + di, j + dj
                            # Verificar que los vecinos estén dentro de los límites de la imagen
                            if 0 <= ni < img_height and 0 <= nj < img_width:
                                # Si el SE permite esa dirección y el vecino no está marcado
                                if SE[di + center_x, dj + center_y] == 1 and outImage[ni, nj] == 0:
                                    outImage[ni, nj] = 1  # Marcar el vecino en `outImage`
                                    changed = True

    return outImage


# ----------------------------------------- DETECCION DE BORDES  -----------------------------------------


def gradientImage(inImage, operator):
    # Definir los kernels de cada operador
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
        Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    else:
        raise ValueError("Operador no válido. Use 'Roberts', 'CentralDiff', 'Prewitt' o 'Sobel'.")

    # Calcular las componentes del gradiente usando filterImage
    gx = filterImage(inImage, Gx)
    gy = filterImage(inImage, Gy)

    return [gx, gy]


def zeroCrossingDetection(image, threshold):
    img_height, img_width = image.shape
    zero_crossing_image = np.zeros_like(image, dtype=np.uint8)

    for i in range(1, img_height - 1):
        for j in range(1, img_width - 1):
            # Verificar si el píxel en (i, j) es menor que -threshold
            if image[i, j] < -threshold:
                # Verificar si alguno de sus ocho vecinos es mayor que threshold
                if (image[i-1:i+2, j-1:j+2] > threshold).any():
                    zero_crossing_image[i, j] = 1  # Marcar como cruce por cero

    return zero_crossing_image


def LoG(inImage, sigma):

    smoothed_image = gaussianFilter(inImage, sigma)
    laplacian_kernel = np.array([[0, -1, 0],
                                 [-1, 4, -1],
                                 [0, -1, 0]])
    masked_image = filterImage(smoothed_image, laplacian_kernel)
    outImage = zeroCrossingDetection(masked_image, 0.01)
    
    return outImage


# ----------------------------------------- SUPRESIÓN NO MÁXIMA -----------------------------------------


def nonMaximumSuppression(magnitude, gradient_direction):
    img_height, img_width = magnitude.shape
    suppressed_image = np.zeros((img_height, img_width), dtype=np.float32)
    # Normalizar la dirección del gradiente a [0, 180)
    gradient_direction = gradient_direction % 180
    
    for i in range(1, img_height - 1):
        for j in range(1, img_width - 1):
            angle = gradient_direction[i, j]
            current_magnitude = magnitude[i, j]
            
            # Comparar con el píxel en la dirección del gradiente
            if (0 <= angle < 22.5) or (157.5 <= angle < 180): # 0 o 180 vecinos en los lados izquierdo y derecho
                neighbor_1 = magnitude[i, j - 1]
                neighbor_2 = magnitude[i, j + 1]
            elif 22.5 <= angle < 67.5: # 45 vecinos en las posiciones diagonal superior derecha e inferior izquierda
                neighbor_1 = magnitude[i - 1, j + 1]
                neighbor_2 = magnitude[i + 1, j - 1]
            elif 67.5 <= angle < 112.5: # 90 vecinos arriba y abajo
                neighbor_1 = magnitude[i - 1, j]
                neighbor_2 = magnitude[i + 1, j]
            else:  # 112.5 <= angle < 157.5 135  vecinos en las posiciones diagonal superior izquierda e inferior derecha
                neighbor_1 = magnitude[i - 1, j - 1]
                neighbor_2 = magnitude[i + 1, j + 1]

            # Suprimir píxeles no máximos
            if current_magnitude >= neighbor_1 and current_magnitude >= neighbor_2:
                suppressed_image[i, j] = current_magnitude

    return suppressed_image

# ----------------------------------------- HISTÉRESIS -----------------------------------------


def hysteresis(suppressed_image, gradient_direction, tlow, thigh):
    # Inicializar la imagen de salida con ceros (sin bordes)
    output_image = np.zeros_like(suppressed_image)
    
    # Marcar los bordes fuertes con 1.0
    strong_edges = suppressed_image >= thigh
    output_image[strong_edges] = 1.0
    
    # Crear una imagen de visita para evitar recorrer puntos ya procesados
    visited = np.zeros_like(suppressed_image, dtype=bool)
    
    # Recorrer la imagen para encontrar bordes fuertes y rastrear hacia los débiles
    img_height, img_width = suppressed_image.shape
    
    # Normalizar la dirección del gradiente a [0, 180)
    gradient_direction = gradient_direction % 180
    
    # Función para rastrear el borde en función de la dirección del gradiente
    def trace_edge(i, j, angle):
        visited[i, j] = True
        output_image[i, j] = 1.0  # Marcar el borde como fuerte
        
        # Comparar con los vecinos según la dirección del gradiente
        if (0 <= angle < 22.5) or (157.5 <= angle < 180):  # 0 o 180 grados: izquierda y derecha
            neighbor_1 = suppressed_image[i, j - 1]
            neighbor_2 = suppressed_image[i, j + 1]
        elif 22.5 <= angle < 67.5:  # 45 grados: diagonal superior derecha e inferior izquierda
            neighbor_1 = suppressed_image[i - 1, j + 1]
            neighbor_2 = suppressed_image[i + 1, j - 1]
        elif 67.5 <= angle < 112.5:  # 90 grados: arriba y abajo
            neighbor_1 = suppressed_image[i - 1, j]
            neighbor_2 = suppressed_image[i + 1, j]
        else:  # 112.5 <= angle < 157.5: diagonal superior izquierda e inferior derecha
            neighbor_1 = suppressed_image[i - 1, j - 1]
            neighbor_2 = suppressed_image[i + 1, j + 1]
        
        # Verificar si el valor del píxel es el mayor de los vecinos
        if suppressed_image[i, j] >= max(neighbor_1, neighbor_2):
            # Si es el mayor, rastrear en esa dirección
            for offset in [-1, 1]:
                ni, nj = i, j
                while 0 <= ni < img_height and 0 <= nj < img_width and suppressed_image[ni, nj] >= tlow and not visited[ni, nj]:
                    visited[ni, nj] = True
                    output_image[ni, nj] = 1.0
                    ni += offset if angle in [0, 180] else 0
                    nj += offset if angle in [90, 270] else 0
        else:
            output_image[i, j] = 0  # Si no es el mayor, suprimir el borde

    # Buscar bordes fuertes y rastrear los débiles conectados
    for i in range(1, img_height - 1):
        for j in range(1, img_width - 1):
            if suppressed_image[i, j] > thigh and not visited[i, j]:
                # Localizamos un borde fuerte
                angle = gradient_direction[i, j]  # Obtener el ángulo del gradiente para (i, j)
                trace_edge(i, j, angle)
    
    return output_image



# ----------------------------------------- DETECTOR DE BORDES CANNY -----------------------------------------

def edgeCanny(inImage, sigma, tlow, thigh):
    # Paso 1: Aplicar suavizado gaussiano
    smoothed_image = gaussianFilter(inImage, sigma)
    cv2.imshow("Suavizado gaussiano", smoothed_image)
    cv2.waitKey(0)

    # Paso 2: Calcular el gradiente usando el operador Sobel
    gx, gy = gradientImage(smoothed_image, "Sobel")
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    gradient_direction = np.degrees(np.arctan2(gy, gx))  # Convertir a grados
    cv2.imshow("Magnitud del gradiente con Sobel", magnitude)
    cv2.waitKey(0)

    # Paso 3: Supresión no máxima
    suppressed_image = nonMaximumSuppression(magnitude, gradient_direction)
    cv2.imshow("Supresion no maxima", suppressed_image)
    cv2.waitKey(0)

    # Paso 4: Umbralización con histéresis
    final_image = hysteresis(suppressed_image, gradient_direction, tlow, thigh)
    cv2.imshow("Histeresis", final_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return final_image
