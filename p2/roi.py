import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import closing, disk, opening
from skimage.feature import canny
from skimage.segmentation import morphological_chan_vese


def get_roi(inImage, window_size):
    """
    Detects the optic disc centroid and extracts the region of interest (ROI) around it.
    Returns the ROI image, relative coordinates, and full pixel coordinates of the ROI.

    Args:
        inImage (numpy.ndarray): Input image in BGR format.

    Returns:
        tuple:
            - numpy.ndarray: Cropped ROI image around the optic disc centroid.
            - tuple: (x_start, y_start, width, height) of the ROI.
            - list: List of (x, y) coordinates of all pixels in the ROI.
    """

    # Extract the green channel and smooth the image
    inImageCopy = inImage.copy()
    green_channel = inImageCopy[:, :, 1]
    smoothed_channel = cv2.GaussianBlur(green_channel, (5, 5), 0)

    # Find the brightest point (initial approximation for ROI center)
    _, _, _, max_loc = cv2.minMaxLoc(smoothed_channel)

    # Define initial ROI for centroid calculation
    x_start = max(max_loc[0] - window_size // 2, 0)
    y_start = max(max_loc[1] - window_size // 2, 0)
    x_end = min(x_start + window_size, inImage.shape[1])
    y_end = min(y_start + window_size, inImage.shape[0])

    # Calculate relative and complete coordinates
    width = x_end - x_start
    height = y_end - y_start
    roi_relative_coords = (x_start, y_start, width, height)

    return roi_relative_coords

def get_roi_from_disc(inImage, disc_roi, window_size):
    """
    Extracts the region of interest (ROI) around the optic disc from the original image.

    Args:
        inImage (numpy.ndarray): Input image in BGR format.
        disc_roi (numpy.ndarray): Binary mask of the optic disc.
        window_size (int): Size of the window to extract around the optic disc.

    Returns:
        tuple:
            - numpy.ndarray: Cropped ROI image around the optic disc.
            - tuple: (x_start, y_start, width, height) of the ROI.
            - list: List of (x, y) coordinates of all pixels in the ROI.
    """
    # Find the bounding box of the disc ROI
    contours, _ = cv2.findContours(disc_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the disc ROI.")
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)

    # Define the ROI around the bounding box
    x_start = max(x - (window_size - w) // 2, 0)
    y_start = max(y - (window_size - h) // 2, 0)
    x_end = min(x_start + window_size, inImage.shape[1])
    y_end = min(y_start + window_size, inImage.shape[0])

    # Calculate relative and complete coordinates
    width = x_end - x_start
    height = y_end - y_start
    roi_relative_coords = (x_start, y_start, width, height)

    return roi_relative_coords


def extract_roi(image, roi_relative_coords):
    """
    Extracts a region of interest (ROI) from an image based on relative coordinates.

    Args:
        image (numpy.ndarray): Input image.
        roi_relative_coords (tuple): (x_start, y_start, width, height) of the ROI.

    Returns:
        numpy.ndarray: Cropped ROI image.
    """
    x_start, y_start, width, height = roi_relative_coords
    x_end = x_start + width
    y_end = y_start + height

    roi = image[y_start:y_end, x_start:x_end]

    return roi



def remove_vessels_disc(image, threshold_low, openSESize):
    """
    Removes blood vessels from the optic disc to improve segmentation.

    Args:
        image (numpy.ndarray): Input image in BGR format.

    Returns:
        numpy.ndarray: Binary edge mask without blood vessels.
    """
    # Extract the red channel to focus on the disc
    red_channel = image[:, :, 2]  # Use the red channel in openCV BGR

    # Apply morphological closing for vessels and white regions
    structuring_element = disk(20)
    closed_image = closing(red_channel, structuring_element).astype(np.uint8)

    # Apply CLAHE to the closed image
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5, 5))
    clahe_image = clahe.apply(closed_image)

    # Detect edges using Canny
    edges = canny(clahe_image, sigma=1.8) 
    edges = (edges * 255).astype(np.uint8)
    
    # Plot the results for visualization
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(red_channel, cmap='gray')
    axes[1].set_title('Red Channel')
    axes[1].axis('off')

    axes[2].imshow(closed_image, cmap='gray')
    axes[2].set_title('Closed Image')
    axes[2].axis('off')

    axes[3].imshow(clahe_image, cmap='gray')
    axes[3].set_title('CLAHE Image')
    axes[3].axis('off')

    axes[4].imshow(edges, cmap='gray')
    axes[4].set_title('Edges')
    axes[4].axis('off')

    plt.show()

    return edges


def remove_vessels_cup(image, disc_mask, threshold_low, threshold_lower, openSESize):
    """
    Removes blood vessels from the optic disc and highlights the cup for segmentation.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        threshold_low (int): Lower threshold for bright regions.

    Returns:
        numpy.ndarray: Binary edge mask without blood vessels and highlighting the cup.
    """

    masked_image = cv2.bitwise_and(image, image, mask=disc_mask)

    # 1. Extraer el canal verde
    green_channel = masked_image[:, :, 1] if image.ndim == 3 else image

    # 4. Aplicar CLAHE a las zonas claras
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(green_channel)

    # 5. Umbralización para resaltar la copa
    _, thresholded_cup = cv2.threshold(clahe_image, threshold_lower, 255, cv2.THRESH_BINARY)

    # Morphological opening to remove small gray spots
    structuring_element = disk(openSESize)  # Disk size depends on vessel size
    opened_image = opening(thresholded_cup, structuring_element).astype(np.uint8)

    structuring_element = disk(openSESize)  # Disk size depends on vessel size
    closed_image = closing(opened_image, structuring_element).astype(np.uint8)


    # Visualización de resultados
    # fig, axes = plt.subplots(1, 6, figsize=(25, 5))
    # axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # axes[0].set_title('Imagen Original')
    # axes[0].axis('off')

    # axes[1].imshow(green_channel, cmap='gray')
    # axes[1].set_title('Canal Verde')
    # axes[1].axis('off')

    # axes[2].imshow(clahe_image, cmap='gray')
    # axes[2].set_title('CLAHE aplicado')
    # axes[2].axis('off')

    # axes[3].imshow(thresholded_cup, cmap='gray')
    # axes[3].set_title('Umbral para la Copa')
    # axes[3].axis('off')

    # axes[4].imshow(opened_image, cmap='gray')
    # axes[4].set_title('Open')
    # axes[4].axis('off')

    # axes[5].imshow(closed_image, cmap='gray')
    # axes[5].set_title('Closed')
    # axes[5].axis('off')

    # plt.show()

    return opened_image


def segment_disc(original_image, roi_relative_coords, edges, image_name):
    """
    Segments the optic disc using morphological snakes and ellipse fitting.
    Maps the result back to the original image dimensions using relative coordinates.

    Args:
        original_image (numpy.ndarray): Original image.
        roi_relative_coords (tuple): (x_start, y_start, width, height) of the ROI.
        edges (numpy.ndarray): Edges generated by Canny in the ROI.
        image_name (str): Name for visualization.

    Returns:
        numpy.ndarray: Optic disc mask mapped to the original image dimensions.
    """
    # Inicializar el nivel de contorno circular en la ROI
    init = np.zeros_like(edges)
    cx, cy = edges.shape[1] // 2, edges.shape[0] // 2
    r = min(cx, cy) - 15
    cv2.circle(init, (cx, cy), r, 1, -1)

    # Evolución de la snake con Morphological Chan-Vese
    snake = morphological_chan_vese(edges, num_iter=66, init_level_set=init, smoothing=1)

    # Encontrar contornos en la snake
    contours, _ = cv2.findContours(snake.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    # Ajustar un elipse al contorno más grande
    ellipse = cv2.fitEllipse(max_contour)

    # Crear una máscara binaria del disco óptico en la ROI
    mask_roi = np.zeros_like(edges, dtype=np.uint8)
    cv2.ellipse(mask_roi, ellipse, (255, 255, 255), -1)

    # Mapear la máscara de la ROI a la imagen original
    x_start, y_start, width, height = roi_relative_coords
    mask_full = np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.uint8)
    mask_full[y_start:y_start + height, x_start:x_start + width] = mask_roi

    # Visualización de pasos intermedios usando matplotlib
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(image_name, fontsize=16)

    axes[0].imshow(snake, cmap='gray')
    axes[0].set_title('Snake aplicado')
    axes[0].axis('off')

    axes[1].imshow(mask_roi, cmap='gray')
    axes[1].set_title('Máscara en ROI')
    axes[1].axis('off')

    axes[2].imshow(mask_full, cmap='gray')
    axes[2].set_title('Máscara en Imagen Completa')
    axes[2].axis('off')

    axes[3].imshow(original_image)
    axes[3].set_title('Imagen Original')
    axes[3].axis('off')

    axes[4].imshow(mask_full, alpha=0.5, cmap='gray')
    axes[4].imshow(original_image, alpha=0.5)
    axes[4].set_title('Máscara Superpuesta')
    axes[4].axis('off')

    plt.show()

    return mask_full




def segment_cup(original_image, roi_relative_coords, edges, image_name):
    """
    Segments the optic cup using morphological snakes and ellipse fitting.
    Maps the result back to the original image dimensions using relative coordinates.

    Args:
        original_image (numpy.ndarray): Original image in BGR format.
        edges (numpy.ndarray): Edge-detected image of the ROI.
        roi_relative_coords (tuple): (x_start, y_start, width, height) of the ROI.
        image_name (str): Name for visualization.

    Returns:
        numpy.ndarray: Binary mask of the optic cup mapped to the original image.
    """

    # Encontrar contornos y ajustar elipse
    contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Combine all contours to fit an ellipse around all white regions
    all_contours = np.vstack(contours)
    ellipse = cv2.fitEllipse(all_contours)

    # Crear máscara binaria de la copa óptica en la ROI
    mask_roi = np.zeros_like(edges, dtype=np.uint8)
    cv2.ellipse(mask_roi, ellipse, 255, -1)

    # Mapear la máscara de la ROI a la imagen completa
    x_start, y_start, width, height = roi_relative_coords
    mask_full = np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.uint8)
    mask_full[y_start:y_start + height, x_start:x_start + width] = mask_roi

    # Dibujar la elipse en la imagen completa
    result_full = original_image.copy()
    ellipse_mapped = ((ellipse[0][0] + x_start, ellipse[0][1] + y_start), ellipse[1], ellipse[2])
    cv2.ellipse(result_full, ellipse_mapped, (0, 255, 0), 2)

    # Visualización
    # fig, axes = plt.subplots(1, 4, figsize=(25, 5))

    # axes[0].imshow(mask_roi, cmap='gray')
    # axes[0].set_title('Máscara en ROI')
    # axes[0].axis('off')

    # axes[1].imshow(mask_full, cmap='gray')
    # axes[1].set_title('Máscara en imagen completa')
    # axes[1].axis('off')

    # axes[2].imshow(cv2.cvtColor(result_full, cv2.COLOR_BGR2RGB))
    # axes[2].set_title('Copa en imagen completa')
    # axes[2].axis('off')

    # axes[3].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    # axes[3].imshow(mask_full, alpha=0.5, cmap='gray')
    # axes[3].set_title('Máscara superpuesta')
    # axes[3].axis('off')

    plt.show()

    return mask_full




# def segment_cup(roi, adjusted_image, image_name):
#     """
#     Segments the optic cup within the ROI.

#     Args:
#         roi (numpy.ndarray): Region of interest image.
#         adjusted_image (numpy.ndarray): Image with adjusted contours.
#         image_name (str): Name for visualization.

#     Returns:
#         numpy.ndarray: Optic cup mask.
#     """
#     # Canal verde invertido y mejorado
#     green_channel = roi[:, :, 1] if roi.ndim == 3 else roi
#     inverted_green = 255 - green_channel
#     enhanced_green = cv2.GaussianBlur(inverted_green, (5, 5), 0)
#     enhanced_green = cv2.addWeighted(enhanced_green, 0.0, inverted_green, 2, 0)

#     # Operaciones morfológicas
#     structuring_element = disk(15)
#     closed_image = opening(enhanced_green, structuring_element).astype(np.uint8)

#     # Umbralización y detección de bordes
#     thresholded = cv2.adaptiveThreshold(closed_image, 200, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 33, 50)
#     thresholded = cv2.bitwise_not(thresholded)
#     edges = canny(thresholded, sigma=2.3)
#     edges = (edges * 255).astype(np.uint8)
#     closed_edges = closing(edges, disk(12))

#     # Encontrar el contorno más grande
#     contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     max_contour = max(contours, key=cv2.contourArea)

#     # Crear una máscara de la copa óptica
#     mask_cup = np.zeros_like(roi)
#     M = cv2.moments(max_contour)
#     if M["m00"] != 0:
#         cx = int(M["m10"] / M["m00"])
#         cy = int(M["m01"] / M["m00"])
#         distances = [np.sqrt((cx - pt[0][0])**2 + (cy - pt[0][1])**2) for pt in max_contour]
#         max_distance = max(distances)

#         if max_distance > 34:
#             ellipse = cv2.fitEllipse(max_contour)
#             cv2.ellipse(adjusted_image, ellipse, (0, 255, 255), 1)
#             cv2.ellipse(mask_cup, ellipse, (255, 255, 255), -1)
#         else:
#             cv2.circle(adjusted_image, (cx, cy), int(max_distance), (0, 255, 255), 1)
#             cv2.circle(mask_cup, (cx, cy), int(max_distance), (255, 255, 255), -1)

#     mask_cup = cv2.cvtColor(mask_cup, cv2.COLOR_BGR2GRAY)

#     # Visualización de pasos intermedios usando matplotlib
#     fig, axes = plt.subplots(1, 4, figsize=(20, 5))
#     fig.suptitle(image_name, fontsize=16)

#     axes[0].imshow(inverted_green, cmap='gray')
#     axes[0].set_title('Canal verde invertido')
#     axes[0].axis('off')

#     axes[1].imshow(enhanced_green, cmap='gray')
#     axes[1].set_title('Realce')
#     axes[1].axis('off')

#     axes[2].imshow(closed_edges, cmap='gray')
#     axes[2].set_title('Bordes detectados')
#     axes[2].axis('off')

#     axes[3].imshow(mask_cup, cmap='gray')
#     axes[3].set_title('Máscara final de la copa')
#     axes[3].axis('off')

#     plt.show()

#     return mask_cup