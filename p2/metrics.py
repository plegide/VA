import numpy as np


def compute_iou(mask1, mask2):
    """
    Compute Intersection over Union (IoU) between two binary masks.

    Args:
        mask1 (numpy.ndarray): First binary mask.
        mask2 (numpy.ndarray): Second binary mask.

    Returns:
        float: IoU metric.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0

def compute_dice(mask1, mask2):
    """
    Compute Dice coefficient between two binary masks.

    Args:
        mask1 (numpy.ndarray): First binary mask.
        mask2 (numpy.ndarray): Second binary mask.

    Returns:
        float: Dice coefficient.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    return (2 * intersection) / (mask1.sum() + mask2.sum()) if (mask1.sum() + mask2.sum()) != 0 else 0
