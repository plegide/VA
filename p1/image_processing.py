import numpy as np
from collections import deque
import matplotlib.pyplot as plt


# ----------------------------------------- HISTOGRAMAS: MEJORA DE CONTRASTE -----------------------------------------


def plot_comparison_histograms(original_image, equalized_image, nBins=256):
    '''
    Plotea los histogramas de la imagen original y la procesada en subplots separados.
    '''
    
    plt.figure(figsize=(12, 6))
    plt.suptitle("Histogramas de las Imágenes")
    
    # Histograma de la imagen original
    plt.subplot(1, 2, 1)
    plt.hist(original_image.flatten(), bins=nBins, color='blue', alpha=0.7)
    plt.title("Histograma Original")
    plt.xlabel("Intensidad")
    plt.ylabel("Frecuencia")
    plt.grid(alpha=0.3)
    
    # Histograma de la imagen ecualizada
    plt.subplot(1, 2, 2)
    plt.hist(equalized_image.flatten(), bins=nBins, color='green', alpha=0.7)
    plt.title("Histograma Procesado")
    plt.xlabel("Intensidad")
    plt.ylabel("Frecuencia")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()



def adjustIntensity(inImage, inRange=[], outRange=[0, 1]):
    '''
    outImage = adjustIntensity (inImage, inRange=[], outRange=[0 1])
    
        inImage: Matriz MxN con la imagen de entrada.
        outImage: Matriz MxN con la imagen de salida.
        inRange: Vector 1x2 con el rango de niveles de intensidad [imin, imax] de entrada.
            Si el vector está vacío (por defecto), el mínimo y máximo de la imagen de entrada
            se usan como imin e imax.
        outRange: Vector 1x2 con el rango de niveles de instensidad [omin, omax] de salida.
            El valor por defecto es [0 1].
    '''

    if len(inRange) == 0: # Si no se especifica el rango de entrada, se saca de la imagen
        imin, imax = np.min(inImage), np.max(inImage)
    else:
        imin, imax = inRange

    omin, omax = outRange

    outImage = omin + (inImage - imin) * (omax - omin) / (imax - imin)

    plot_comparison_histograms(inImage, outImage)
    
    return outImage


def equalizeIntensity(inImage, nBins=256):
    '''
    outImage = equalizeIntensity (inImage, nBins=256)

        inImage, outImage: ...
        nBins: Número de bins utilizados en el procesamiento. Se asume que el intervalo de
            entrada [0 1] se divide en nBins intervalos iguales para hacer el procesamiento,
            y que la imagen de salida vuelve a quedar en el intervalo [0 1]. Por defecto 256.
    '''

    M, N = inImage.shape
    H = np.zeros(nBins) # Histograma de la imagen
    for i in range(M):
        for j in range(N):
            pixel_value = int(inImage[i, j] * (nBins - 1))  # Mapear de [0, 1] a [0, nBins - 1]
            H[pixel_value] += 1  # Se aumenta el bin del histograma para ese valor de pixel

    Hc = H.cumsum() # Histograma acumulado
    T = (Hc / (M*N)) * 255 # Funcion de transformacion

    # Mapear cada píxel de la imagen original a partir de T
    outImage = np.zeros_like(inImage)
    for i in range(M):
        for j in range(N):
            pixel_value = int(inImage[i, j] * (nBins - 1))
            outImage[i, j] = T[pixel_value] / 255.0  # Normalizar de nuevo a [0, 1]
    
    plot_comparison_histograms(inImage, outImage)

    return outImage


# ----------------------------------------- FILTRADO ESPACIAL: SUAVIZADO -----------------------------------------


def filterImage(inImage, kernel):
    '''
    outImage = filterImage (inImage, kernel)

        inImage, outImage: ...
        kernel: Matriz PxQ con el kernel del filtro de entrada. Se asume que la posición central
            del filtro está en (⌊P/2⌋ + 1, ⌊Q/2⌋ + 1).
    '''

    kernel = np.array(kernel)
    img_height, img_width = inImage.shape
    k_height, k_width = kernel.shape
    pad_height = k_height // 2 # Division entera para kernels de tamaño impar
    pad_width = k_width // 2
    outImage = np.zeros_like(inImage) 
    padded_image = np.pad(inImage, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')

    for i in range(img_height): # Convolucion con el filtro
        for j in range(img_width):
            region = padded_image[i:i + k_height, j:j + k_width] # Extraer la región correspondiente de la imagen
            outImage[i, j] = np.sum(region * kernel) # Calcular la convolución y asignarla a la imagen de salida

    return outImage


def plotGaussKernel(x, kernel, sigma):
    '''
    Genera el gráfico de la campana de Gauss a partir de un kernel.
    '''

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
    '''
    kernel = gaussKernel1D (sigma)

        sigma: Parámetro σ de entrada.
        kernel: Vector 1xN con el kernel de salida, teniendo en cuenta que:
            • El centro x = 0 de la Gaussiana está en la posición ⌊N/2⌋ + 1.
            • N se calcula a partir de σ como N = 2⌈3σ⌉ + 1.
    '''

    N = int(2 * np.ceil(3 * sigma) + 1)  # Calcular N a partir de sigma
    x = np.linspace(-(N // 2), N // 2, N)  # Crear de valores simétricos al rededor e de 0
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))  # Calcular el kernel gaussiano
    kernel /= np.sum(kernel)  # Normalizar el kernel

    # Plotear la campana de gauss
    plotGaussKernel(x, kernel, sigma)

    return kernel


def gaussianFilter(inImage, sigma):
    '''
    outImage = gaussianFilter (inImage, sigma)

        inImage, outImage, sigma: ...

        NOTA. Como el filtro Gaussiano es lineal y separable podemos implementar este suavi-
        zado simplemente convolucionando la imagen, primero, con un kernel Gaussiano unidi-
        mensional 1×N y, luego, convolucionando el resultado con el kernel transpuesto N×1.
    '''

    kernel_1d = gaussKernel1D(sigma)
    temp_image = filterImage(inImage, kernel_1d.reshape(1, -1))  # Aplicar como filtro 1xN
    outImage = filterImage(temp_image, kernel_1d.reshape(-1, 1))  # Aplicar como filtro Nx1 traspuesto

    return outImage


def medianFilter(inImage, filterSize):
    '''
    outImage = medianFilter (inImage, filterSize)
    
        inImage, outImage: ...
        filterSice: Valor entero N indicando que el tamaño de ventana es de NxN. La posición
            central de la ventana es (⌊N/2⌋ + 1, ⌊N/2⌋ + 1).
    '''

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
    '''
    outImage = erode (inImage, SE, center=[])

        inImage, outImage: ...
        SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
        center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es
            la esquina superior izquierda. Si es un vector vacío (valor por defecto), el centro
            se calcula como (⌊P/2⌋ + 1, ⌊Q/2⌋ + 1)
    '''

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
    '''
    outImage = dilate (inImage, SE, center=[])

        inImage, outImage: ...
        SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
        center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es
            la esquina superior izquierda. Si es un vector vacío (valor por defecto), el centro
            se calcula como (⌊P/2⌋ + 1, ⌊Q/2⌋ + 1)
    '''

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
    '''
    outImage = opening (inImage, SE, center=[])

        inImage, outImage: ...
        SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
        center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es
            la esquina superior izquierda. Si es un vector vacío (valor por defecto), el centro
            se calcula como (⌊P/2⌋ + 1, ⌊Q/2⌋ + 1)
    '''

    return dilate(erode(inImage, SE, center), SE, center) 


def closing(inImage, SE, center=[]):
    '''
    outImage = closing (inImage, SE, center=[])

        inImage, outImage: ...
        SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
        center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es
            la esquina superior izquierda. Si es un vector vacío (valor por defecto), el centro
            se calcula como (⌊P/2⌋ + 1, ⌊Q/2⌋ + 1)
    '''
    
    return erode(dilate(inImage, SE, center), SE, center)


def fill(inImage, seeds, SE=[], center=[]):
    """
    outImage = fill (inImage, seeds, SE=[], center=[])

        inImage, outImage, center: ...
        seeds: Matriz Nx2 con N coordenadas (fila,columna) de los puntos semilla.
        SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante de conectividad.
            Si es un vector vacío se asume conectividad 4 (cruz 3 × 3).
    """

    # Si no se pasa un SE, 4-vecinos
    if (isinstance(SE, list) and len(SE) == 0) or (isinstance(SE, np.ndarray) and SE.size == 0):
        SE = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=bool)
        
    fill = np.zeros_like(inImage, dtype=bool)
    complementary = np.logical_not(inImage) 
    for seed in seeds:
        fill[seed[0], seed[1]] = 1  #Poner los puntos semilla a 1

    while True: # Dilatacion condicional
        previous = fill.copy()
        dilated = dilate(previous, SE, center) # Aqui se cubre el caso del centro vacio
        fill = np.logical_and(dilated, complementary) # Interseccion con el complementario
        if np.array_equal(previous, fill): # Si no se generan nuevos 1s se para
            break

    outImage = np.logical_or(fill, inImage) # Union del relleno y la imagen original

    return outImage


# ----------------------------------------- DETECCION DE BORDES  -----------------------------------------


def gradientImage(inImage, operator):
    '''
    [gx, gy] = gradientImage (inImage, operator)

        inImage: ...
        gx, gy: Componentes Gx y Gy del gradiente.
        operator: Permite seleccionar el operador utilizado mediante los valores: ’Roberts’,
            ’CentralDiff’, ’Prewitt’ o ’Sobel’.
    '''

    if operator == 'Roberts':
        Gx = np.array([[1, 0], [0, -1]])
        Gy = Gx.T

    elif operator == 'CentralDiff':
        Gx = np.array([[-1, 0, 1]])
        Gy = Gx.T

    elif operator == 'Prewitt':
        Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        Gy = Gx.T

    elif operator == 'Sobel':
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Gy = Gx.T

    gx = filterImage(inImage, Gx)
    gy = filterImage(inImage, Gy)
    
    return [gx, gy]


def LoG(inImage, sigma):
    '''
    outImage = LoG (inImage, sigma)

        inImage, outImage: ...
        sigma: Parámetro σ de la Gaussiana.
    '''

    N = int(2 * np.ceil(3 * sigma) + 1)  # Tamaño del kernel en funcion de sigma
    kernel = np.zeros((N, N))
    center = N // 2

    for x in range(N):
        for y in range(N):
            x_rel = x - center 
            y_rel = y - center
            kernel[x, y] = ((x_rel**2 + y_rel**2 - sigma**2) / sigma**4) * np.exp(-(x_rel**2 + y_rel**2) / (2 * sigma**2)) # Laplaciano de la Gaussiana

    kernel = kernel - np.mean(kernel)  # Normalizar
    outImage = filterImage(inImage, kernel)
    
    return outImage


def nonMaximumSuppression(magnitude, gradient_direction):
    '''
    suppressed_image = nonMaximumSuppression (magnitude, gradient_direction)

        magnitude: Matriz MxN con la magnitud del gradiente.
        gradient_direction: Matriz MxN con la dirección del gradiente.
        suppressed_image: Matriz MxN con la imagen tras la supresión de no-máximos.
    '''

    img_height, img_width = magnitude.shape
    suppressed_image = np.zeros((img_height, img_width), dtype=np.float32)
    gradient_direction = gradient_direction % 180 # Normalizar el angulo a [0,180)
    
    for i in range(1, img_height - 1):
        for j in range(1, img_width - 1):
            angle = gradient_direction[i, j]
            current_magnitude = magnitude[i, j]   
            # Los vecinos son los pixeles en la direccion del gradiente
            if (0 <= angle < 22.5) or (157.5 <= angle < 180): # vecinos en los lados izquierdo y derecho
                neighbor_1 = magnitude[i, j - 1]
                neighbor_2 = magnitude[i, j + 1]
            elif 22.5 <= angle < 67.5: # vecinos en las posiciones diagonal superior izquierda e inferior derecha
                neighbor_1 = magnitude[i - 1, j - 1]
                neighbor_2 = magnitude[i + 1, j + 1]
            elif 67.5 <= angle < 112.5: # vecinos arriba y abajo
                neighbor_1 = magnitude[i - 1, j]
                neighbor_2 = magnitude[i + 1, j]
            else:  # 112.5 <= angle < 157.5 135 vecinos en las posiciones diagonal superior derecha e inferior izquierda
                neighbor_1 = magnitude[i - 1, j + 1]
                neighbor_2 = magnitude[i + 1, j - 1]

            if current_magnitude >= neighbor_1 and current_magnitude >= neighbor_2: # Si es el maximo entre los vecinos
                suppressed_image[i, j] = current_magnitude

    return suppressed_image


def hysteresis(suppressed_image, gradient_direction, tlow, thigh, neighbor_depth=3):
    '''
    output_image = hysteresis (suppressed_image, gradient_direction, tlow, thigh, neighbor_depth=3)

        suppressed_image: Matriz MxN con la imagen tras la supresión de no-máximos.
        gradient_direction: Matriz MxN con la dirección del gradiente.
        tlow, thigh: Umbrales de histéresis bajo y alto, respectivamente.
        neighbor_depth: Profundidad de los vecinos en la dirección del borde.
        output_image: ...
    '''

    output_image = np.zeros_like(suppressed_image)
    img_height, img_width = suppressed_image.shape
    gradient_direction = gradient_direction % 180  # Normalizar el angulo a [0,180)

    queue = deque()
    for i in range(1, img_height - 1):
        for j in range(1, img_width - 1):
            if suppressed_image[i, j] >= thigh: # Bordes fuertes
                output_image[i, j] = 1.0
                queue.append((i, j))

    while queue: 
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
    '''
    outImage = edgeCanny (inImage, sigma, tlow, thigh)

        inImage, outImage: ...
        sigma: Parámetro σ del filtro Gaussiano.
        tlow, thigh: Umbrales de histéresis bajo y alto, respectivamente.
    '''

    smoothed_image = gaussianFilter(inImage, sigma) #Suavizado gaussiano

    gx, gy = gradientImage(smoothed_image, "Sobel") # Gradiente con Sobel
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    gradient_direction = np.degrees(np.arctan2(gy, gx))

    suppressed_image = nonMaximumSuppression(magnitude, gradient_direction) # Supresión no máxima

    final_image = hysteresis(suppressed_image, gradient_direction, tlow, thigh)

    # Plot de los pasos de Canny
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