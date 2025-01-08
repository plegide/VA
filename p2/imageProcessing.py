import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.segmentation import morphological_chan_vese


def get_roi(input_image, window_size):

    # Extract the green channel and smooth the image
    input_image_copy = input_image.copy()
    green_channel = input_image_copy[:, :, 1]
    smoothed_channel = cv2.GaussianBlur(green_channel, (5, 5), 0)
    _, _, _, centroid = cv2.minMaxLoc(smoothed_channel) # Centroid is the brightest point

    # ROI coordinates around the centroid
    x_start = max(centroid[0] - window_size // 2, 0)
    y_start = max(centroid[1] - window_size // 2, 0)
    x_end = min(x_start + window_size, input_image.shape[1])
    y_end = min(y_start + window_size, input_image.shape[0])

    # Compute relative coordinates to adjust masks to full images later
    width = x_end - x_start
    height = y_end - y_start
    roi_relative_coords = (x_start, y_start, width, height)

    return roi_relative_coords, centroid


def get_roi_from_disc(input_image, disc_roi, window_size):

    # Find the bounding box that contains the disc mask
    contours, _ = cv2.findContours(disc_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)

    # Define the new ROI around the bounding box
    x_start = max(x - (window_size - w) // 2, 0)
    y_start = max(y - (window_size - h) // 2, 0)
    x_end = min(x_start + window_size, input_image.shape[1])
    y_end = min(y_start + window_size, input_image.shape[0])

    # Relative coordinates to adjust the cup mask to the full image later
    width = x_end - x_start
    height = y_end - y_start
    roi_relative_coords = (x_start, y_start, width, height)

    return roi_relative_coords


def extract_roi(input_image, roi_relative_coords):

    x_start, y_start, width, height = roi_relative_coords
    x_end = x_start + width
    y_end = y_start + height

    roi = input_image[y_start:y_end, x_start:x_end] # Extract the ROI from the image

    return roi


def remove_vessels_disc(input_image, closing_se_size, clahe_clip, canny_sigma):

    red_channel = input_image[:, :, 2]  # Use the red channel in openCV BGR

    # Closing for removing vessels and white regions
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_se_size, closing_se_size))
    closed_image = cv2.morphologyEx(red_channel, cv2.MORPH_CLOSE, structuring_element)

    # Apply CLAHE to the closed image
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(5, 5))
    clahe_image = clahe.apply(closed_image)

    # Extract the edges with canny
    edges = canny(clahe_image, sigma=canny_sigma) 
    edges = (edges * 255).astype(np.uint8)
    
    # Mid steps visualization

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    axes[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
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


def remove_vessels_cup(input_image, disc_mask, threshold_lower, close_se_size):

    masked_image = cv2.bitwise_and(input_image, input_image, mask=disc_mask) # Only take into account the disc region

    green_channel = masked_image[:, :, 1] if input_image.ndim == 3 else input_image # Use the green channel in openCV BGR

    # Remove vessels and with closing
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_se_size, close_se_size))
    closed_image = cv2.morphologyEx(green_channel, cv2.MORPH_CLOSE, structuring_element)

    # Threshold to obtain the brighter regions inside the cup
    _, thresholded_cup = cv2.threshold(closed_image, threshold_lower, 255, cv2.THRESH_BINARY)

    # Mid steps visualization

    fig, axes = plt.subplots(1, 4, figsize=(25, 5))
    axes[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(closed_image, cmap='gray')
    axes[1].set_title('Closed Image')
    axes[1].axis('off')

    axes[2].imshow(green_channel, cmap='gray')
    axes[2].set_title('Green Channel')
    axes[2].axis('off')

    axes[3].imshow(thresholded_cup, cmap='gray')
    axes[3].set_title('Thresholded Cup')
    axes[3].axis('off')

    plt.show()

    return thresholded_cup


def segment_disc(input_image, roi_relative_coords, edges):

    # Start a circular contour in the center of the ROI
    init = np.zeros_like(edges)
    cx, cy = edges.shape[1] // 2, edges.shape[0] // 2
    r = min(cx, cy) - 15
    cv2.circle(init, (cx, cy), r, 1, -1)

    # The snake evolves using the Chan-Vese method
    snake = morphological_chan_vese(edges, num_iter=66, init_level_set=init, smoothing=1)

    # Find snake contours once its finished
    contours, _ = cv2.findContours(snake.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    # Adjust an ellipse to the contour
    ellipse = cv2.fitEllipse(max_contour)

    # Draw the binary mask
    mask_roi = np.zeros_like(edges, dtype=np.uint8)
    cv2.ellipse(mask_roi, ellipse, (255, 255, 255), -1)

    # Adjust the mask to the original image size with relative coordinates
    x_start, y_start, width, height = roi_relative_coords
    mask_full = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.uint8)
    mask_full[y_start:y_start + height, x_start:x_start + width] = mask_roi

    # Mid steps visualization

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    axes[0].imshow(snake, cmap='gray')
    axes[0].set_title('Snake Applied')
    axes[0].axis('off')

    axes[1].imshow(mask_roi, cmap='gray')
    axes[1].set_title('Mask in ROI')
    axes[1].axis('off')

    axes[2].imshow(mask_full, cmap='gray')
    axes[2].set_title('Mask in Full Image')
    axes[2].axis('off')

    axes[3].imshow(input_image)
    axes[3].set_title('Original Image')
    axes[3].axis('off')

    axes[4].imshow(mask_full, alpha=0.5, cmap='gray')
    axes[4].imshow(input_image, alpha=0.5)
    axes[4].set_title('Overlay Mask')
    axes[4].axis('off')

    plt.show()

    return mask_full


def segment_cup(input_image, roi_relative_coords, edges):

    # Stack all the available contours and fit an ellipse
    contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = np.vstack(contours)
    ellipse = cv2.fitEllipse(all_contours)

    ellipse_center = (ellipse[0][0] - 15, ellipse[0][1])  # Adjust the ellipse center to the left
    ellipse_size = (ellipse[1][0] + 17, ellipse[1][1] + 5)  # The ellipse is always wider and a bit bigger because of the blood vessels

    # Draw the ellipse as a binary mask
    mask_roi = np.zeros_like(edges, dtype=np.uint8)
    cv2.ellipse(mask_roi, (ellipse_center, ellipse_size, ellipse[2]), 255, -1)

    # Adjust the mask to the image real size with relative coordinates
    x_start, y_start, width, height = roi_relative_coords
    mask_full = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.uint8)
    mask_full[y_start:y_start + height, x_start:x_start + width] = mask_roi

    # Draw the ellipse in the full image
    result_full = input_image.copy()
    ellipse_mapped = ((ellipse_center[0] + x_start, ellipse_center[1] + y_start), ellipse_size, ellipse[2])
    cv2.ellipse(result_full, ellipse_mapped, (0, 255, 0), 2)

    # Mid steps visualization
    
    fig, axes = plt.subplots(1, 4, figsize=(25, 5))

    axes[0].imshow(mask_roi, cmap='gray')
    axes[0].set_title('Mask in ROI')
    axes[0].axis('off')

    axes[1].imshow(mask_full, cmap='gray')
    axes[1].set_title('Mask in Full Image')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(result_full, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Cup in Full Image')
    axes[2].axis('off')

    axes[3].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    axes[3].imshow(mask_full, alpha=0.5, cmap='gray')
    axes[3].set_title('Overlay Mask')
    axes[3].axis('off')

    plt.show()

    return mask_full