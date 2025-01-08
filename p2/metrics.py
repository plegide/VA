import numpy as np


def compute_centroid(binary_mask):

    binary_mask = binary_mask > 0
    indexes = np.argwhere(binary_mask) # Get the indexes of the non-zero elements in binary masks
    centroid = indexes.mean(axis=0)
    centroid = centroid[::-1]  # Coordinates are originally in reverse order so reverse them
    
    return tuple(centroid)


def compute_euclidean_distance(point1, point2):

    point1 = np.array(point1)
    point2 = np.array(point2)

    return np.linalg.norm(point1 - point2) # Euclidean distance between two points


def compute_iou(mask1, mask2):

    mask1 = mask1 > 0
    mask2 = mask2 > 0
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    return intersection / union if union != 0 else 0


def compute_dice(mask1, mask2):

    mask1 = mask1 > 0
    mask2 = mask2 > 0
    intersection = np.logical_and(mask1, mask2).sum()

    return (2 * intersection) / (mask1.sum() + mask2.sum()) if (mask1.sum() + mask2.sum()) != 0 else 0


def compute_cdr(disc_mask, cup_mask):

    # Get the y indexes for the location of both binary masks
    disc_y_indexes = np.where(disc_mask > 0)[0]
    cup_y_indexes = np.where(cup_mask > 0)[0]
    disc_height = disc_y_indexes.max() - disc_y_indexes.min()
    cup_height = cup_y_indexes.max() - cup_y_indexes.min()

    cdr = cup_height / disc_height

    return cdr
