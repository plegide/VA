import os
import cv2


def load_images(image_folder):

    images = []
    filenames = sorted(os.listdir(image_folder))  # Images are always processed in the same order
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            images.append((filename, image))
            
    return images


def load_ground_truth(gt_folder):

    ground_truths = {'disc': {}, 'cup': {}}
    filenames = sorted(os.listdir(gt_folder))
    
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            gt_path = os.path.join(gt_folder, filename)
            gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)  # Images are read in gray

            if 'disc' in filename.lower():
                ground_truths['disc'][filename] = gt_image
            elif 'cup' in filename.lower():
                ground_truths['cup'][filename] = gt_image
    
    return ground_truths
