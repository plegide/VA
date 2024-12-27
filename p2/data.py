import os
import cv2
import numpy as np

def load_images(image_folder):
    """
    Loads images from a folder.

    Args:
        image_folder (str): Path to the folder containing the images.

    Returns:
        list: List of tuples (filename, image) where image is a NumPy array.
    """
    images = []
    filenames = sorted(os.listdir(image_folder))  # Images are always in the same order
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            images.append((filename, image))
    return images


def load_ground_truth(gt_folder):
    """
    Loads ground truth images for optic disc and cup from a folder.

    Args:
        gt_folder (str): Path to the folder containing the ground truth images.

    Returns:
        dict: A dictionary with keys 'disc' and 'cup', each containing a dictionary
              of filenames and corresponding ground truth images as NumPy arrays.
    """
    ground_truths = {'disc': {}, 'cup': {}}
    filenames = sorted(os.listdir(gt_folder))
    
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            gt_path = os.path.join(gt_folder, filename)
            gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
            
            # Categorize the image as 'disc' or 'cup' based on its filename
            if 'disc' in filename.lower():
                ground_truths['disc'][filename] = gt_image
            elif 'cup' in filename.lower():
                ground_truths['cup'][filename] = gt_image
    
    return ground_truths
