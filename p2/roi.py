import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import closing, disk, opening
from skimage.feature import canny
from skimage.segmentation import morphological_chan_vese
    

def detect_disc(image):
    """
    Detects the optic disc centroid using the green channel and smoothing and filtering operations.

    Args:
        image (numpy.ndarray): Input image in BGR format.

    Returns:
        tuple: Coordinates (x, y) of the centroid and the bounding box of the ROI.
    """

    image_copy = image.copy()

    # Extract the green channel and smooth the image
    green_channel = image[:, :, 1] 
    smoothed_channel = cv2.GaussianBlur(green_channel, (5, 5), 0) 
    kernel = np.ones((3, 3), np.float32) / 9
    filtered_channel = cv2.filter2D(smoothed_channel, -1, kernel) 

    # Find the brightest point and adjust the ROI around it
    _, _, _, max_loc = cv2.minMaxLoc(filtered_channel)
    window_size = 250
    x_start = max(max_loc[0] - window_size // 2, 0)
    y_start = max(max_loc[1] - window_size // 2, 0)
    x_end = min(x_start + window_size, image.shape[1])
    y_end = min(y_start + window_size, image.shape[0])

    # Calculate the centroid
    roi = image_copy[y_start:y_end, x_start:x_end]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi_binary = cv2.threshold(roi_gray, 1, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(roi_binary)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = 0, 0

    centroid = (x_start + cx, y_start + cy)
    bounding_box = (x_start, y_start, x_end, y_end)

    return centroid, bounding_box


def extract_roi(image, bounding_box):
    """
    Extracts the region of interest (ROI) based on a bounding box.

    Args:
        image (numpy.ndarray): Input image.
        bounding_box (tuple): Coordinates of the bounding box (x_start, y_start, x_end, y_end).

    Returns:
        numpy.ndarray: Cropped ROI image.
    """
    x_start, y_start, x_end, y_end = bounding_box
    return image[y_start:y_end, x_start:x_end]


def remove_vessels(image):
    """
    Removes blood vessels from the optic disc to improve segmentation.

    Args:
        image (numpy.ndarray): Input image in BGR format.

    Returns:
        numpy.ndarray: Binary edge mask without blood vessels.
    """
    # Extract the red channel to focus on the disc
    red_channel = image[:, :, 0]
    
    # Adjust brightness and contrast to improve detail visibility, especially in bright images
    alpha = 3.0  # Contrast
    beta = 50    # Brightness
    enhanced_image = cv2.convertScaleAbs(red_channel, alpha=alpha, beta=beta)

    # Smooth the red channel with GaussianBlur to reduce noise
    smoothed_red = cv2.GaussianBlur(enhanced_image, (15, 15), 0)
    
    # Enhance contrast by subtracting and weighting the smoothed image
    enhanced_channel = cv2.addWeighted(smoothed_red, 1.5, smoothed_red, -0.5, 0)

    # Morphological closing operation to remove small noise
    structuring_element = disk(15)  # Smaller disk size for better results
    closed_image = closing(enhanced_channel, structuring_element).astype(np.uint8)

    # Apply CLAHE to dynamically enhance contrast
    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(5, 5))
    clahe_image = clahe.apply(closed_image)

    # Edge detection with Canny
    sigma = 2.0 if min(enhanced_image.shape) > 500 else 1.5  # Smaller sigma for small images
    edges = canny(clahe_image, sigma=sigma)
    edges = (edges * 255).astype(np.uint8)

    # Dilate edges for better definition
    kernel = np.ones((1, 5), np.uint8)  # Larger kernel to expand edges
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)

    # Detect circles with Hough Transform to find the optic disc
    circles = cv2.HoughCircles(
        dilated_edges, 
        cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, 
        param1=50, param2=30, minRadius=20, maxRadius=100
    )

    # If circles are detected, highlight the largest one
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(dilated_edges, (x, y), r, (255, 255, 255), 2)
            cv2.rectangle(dilated_edges, (x - r, y - r), (x + r, y + r), (255, 0, 0), 2)
    
    # Plot the results of each step
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(red_channel, cmap='gray')
    axes[1].set_title('Red Channel')
    axes[1].axis('off')

    axes[2].imshow(enhanced_image, cmap='gray')
    axes[2].set_title('Enhanced Image')
    axes[2].axis('off')

    axes[3].imshow(dilated_edges, cmap='gray')
    axes[3].set_title('Dilated Edges')
    axes[3].axis('off')

    plt.show()

    return dilated_edges


def segment_disc(image, edges, image_name):
    """
    Segments the optic disc using snake methods and ellipse fitting.

    Args:
        image (numpy.ndarray): Original image.
        edges (numpy.ndarray): Edges generated by Canny.
        image_name (str): Name for visualization.

    Returns:
        tuple: Optic disc mask and the image with fitted contours.
    """
    init = np.zeros_like(edges)
    cx, cy = image.shape[1] // 2, image.shape[0] // 2
    r = min(cx, cy) - 15
    cv2.circle(init, (cx, cy), r, 1, -1)
    
    # Apply snake (Morphological Chan-Vese)
    snake = morphological_chan_vese(edges, num_iter=66, init_level_set=init, smoothing=1)
    SE = disk(20)
    snake = closing(snake, SE)
    
    # Detect contours
    contours, _ = cv2.findContours(snake.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    # Fit ellipse using the contour
    ellipse = cv2.fitEllipse(max_contour)

    # Calculate the focal points and radii of the ellipse
    center, axes, angle = ellipse
    major_axis, minor_axis = axes  # Major and minor axes
    center_x, center_y = center

    # Create an empty mask for the optic disc
    mask_disc = np.zeros_like(image, dtype=np.uint8)

    # Draw the ellipse on the mask (filled with white)
    cv2.ellipse(mask_disc, ellipse, (255, 255, 255), -1)

    # Convert the mask to grayscale
    mask_disc = cv2.cvtColor(mask_disc, cv2.COLOR_BGR2GRAY)

    # Draw the ellipse on the original image for visualization
    adjusted_image = image.copy()
    cv2.ellipse(adjusted_image, ellipse, (0, 255, 0), 1)

    return mask_disc, adjusted_image


def segment_cup(roi, adjusted_image, image_name):
    """
    Segments the optic cup within the ROI.

    Args:
        roi (numpy.ndarray): Region of interest image.
        adjusted_image (numpy.ndarray): Image with adjusted contours.
        image_name (str): Name for visualization.

    Returns:
        numpy.ndarray: Optic cup mask.
    """
    # Green channel and enhancement
    green_channel = roi[:, :, 1] if roi.ndim == 3 else roi
    inverted_green = 255 - green_channel
    enhanced_green = cv2.GaussianBlur(inverted_green, (5, 5), 0)
    enhanced_green = cv2.addWeighted(enhanced_green, 0.0, inverted_green, 2, 0)

    # Morphological operations
    structuring_element = disk(15)
    closed_image = opening(enhanced_green, structuring_element).astype(np.uint8)

    # Thresholding and edge detection
    thresholded = cv2.adaptiveThreshold(
        closed_image, 200, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize=33, C=50
    )
    thresholded = cv2.bitwise_not(thresholded)
    edges = canny(thresholded, sigma=2.3)
    edges = (edges * 255).astype(np.uint8)
    closed_edges = closing(edges, disk(12))

    # Detect cup contour
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(max_contour)

    # Create optic cup mask
    mask_cup = np.zeros_like(roi)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        distances = [np.sqrt((cx - pt[0][0])**2 + (cy - pt[0][1])**2) for pt in max_contour]
        max_distance = max(distances)

        if max_distance > 34:
            ellipse = cv2.fitEllipse(max_contour)
            cv2.ellipse(adjusted_image, ellipse, (0, 255, 255), 1)
            cv2.ellipse(mask_cup, ellipse, (255, 255, 255), -1)
        else:
            cv2.circle(adjusted_image, (cx, cy), 1, (0, 0, 255), -1)
            cv2.circle(adjusted_image, (cx, cy), int(max_distance), (0, 255, 255), 1)
            cv2.circle(mask_cup, (cx, cy), int(max_distance), (255, 255, 255), -1)
    return cv2.cvtColor(mask_cup, cv2.COLOR_BGR2GRAY)
