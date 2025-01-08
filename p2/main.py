import cv2
import os
import matplotlib.pyplot as plt
from tabulate import tabulate
from data import load_images, load_ground_truth
from imageProcessing import get_roi, get_roi_from_disc, extract_roi, segment_disc, segment_cup, remove_vessels_disc, remove_vessels_cup
from metrics import compute_centroid, compute_euclidean_distance, compute_iou, compute_dice, compute_cdr

def main():
    images = load_images('refuge_images/retinography/')
    ground_truths = load_ground_truth('refuge_images/ground_truth/')
    results = []

    for idx, ((filename, image)) in enumerate(images):
        # Obtain the ground truth masks corresponding to each image
        gt_filenameDisc = f"{filename.split('.')[0]}_disc.png"
        gt_filenameCup = f"{filename.split('.')[0]}_cup.png"
        gt_disc = ground_truths['disc'].get(gt_filenameDisc, None)
        gt_cup = ground_truths['cup'].get(gt_filenameCup, None)

        # Image processing
        roi_coordinates, disc_centroid = get_roi(image, 200)
        roi_disc = extract_roi(image, roi_coordinates)

        vessels_removed_disc = remove_vessels_disc(roi_disc, 20, 2, 1.8)
        disc_mask = segment_disc(image, roi_coordinates, vessels_removed_disc)

        small_roi_coordinates = get_roi_from_disc(image, disc_mask, 130)
        roi_disc_mask = extract_roi(disc_mask, small_roi_coordinates)
        roi_cup = extract_roi(image, small_roi_coordinates)
        vessels_removed_cup = remove_vessels_cup(roi_cup, roi_disc_mask, 160, 5)
        cup_mask = segment_cup(image, small_roi_coordinates, vessels_removed_cup)

        # Save masks as files
        os.makedirs('result/disc', exist_ok=True)
        os.makedirs('result/cup', exist_ok=True)
        disc_filename = f"result/disc/{filename.split('.')[0]}_disc.png"
        cup_filename = f"result/cup/{filename.split('.')[0]}_cup.png"
        cv2.imwrite(disc_filename, disc_mask)
        cv2.imwrite(cup_filename, cup_mask)
        
        # Compute metrics
        gt_centroid = compute_centroid(gt_disc)
        distance = compute_euclidean_distance(disc_centroid, gt_centroid)

        iou_disc = compute_iou(disc_mask, gt_disc) 
        dice_disc = compute_dice(disc_mask, gt_disc)
        iou_cup = compute_iou(cup_mask, gt_cup)
        dice_cup = compute_dice(cup_mask, gt_cup)

        cdr = compute_cdr(disc_mask, cup_mask)
        gt_cdr = compute_cdr(gt_disc, gt_cup)

        # Show metrics in table format
        results.append([
            filename,
            f"{distance:.4f}",
            f"{iou_disc:.4f}" if iou_disc is not None else "N/A",
            f"{dice_disc:.4f}" if dice_disc is not None else "N/A",
            f"{iou_cup:.4f}" if iou_cup is not None else "N/A",
            f"{dice_cup:.4f}" if dice_cup is not None else "N/A",
            f"{cdr:.4f}",
            f"{gt_cdr:.4f}"
        ])

        # Mid steps visualization
        fig, axes = plt.subplots(1, 6, figsize=(20, 10))
        fig.suptitle(filename)

        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Append the centroid to the ROI image
        roi_image_with_centroid = cv2.cvtColor(roi_disc, cv2.COLOR_BGR2RGB)
        adjusted_centroid = (disc_centroid[0] - roi_coordinates[0], disc_centroid[1] - roi_coordinates[1])
        cv2.circle(roi_image_with_centroid, adjusted_centroid, 5, (255, 0, 0), -1)
        axes[1].imshow(roi_image_with_centroid)
        axes[1].set_title("ROI with Disc Centroid")
        axes[1].axis('off')

        axes[2].imshow(disc_mask, cmap='gray')
        axes[2].set_title("Optic Disc")
        axes[2].axis('off')

        disc_overlapped = cv2.addWeighted(image, 0.7, cv2.cvtColor(disc_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
        axes[3].imshow(cv2.cvtColor(disc_overlapped, cv2.COLOR_BGR2RGB))
        axes[3].set_title("Optic Disc overlapped")
        axes[3].axis('off')

        axes[4].imshow(cup_mask, cmap='gray')
        axes[4].set_title("Cup")
        axes[4].axis('off')

        cup_overlapped = cv2.addWeighted(image, 0.7, cv2.cvtColor(cup_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
        axes[5].imshow(cv2.cvtColor(cup_overlapped, cv2.COLOR_BGR2RGB))
        axes[5].set_title("Cup overlapped")
        axes[5].axis('off')

        plt.tight_layout()
        plt.show()

    # Plot the table with the results
    headers = ["Image", "Disc Localization Error", "Disc IoU", "Disc Dice", "Cup IoU", "Cup Dice", "Image CDR", "Ground Truth CDR"]
    print(tabulate(results, headers=headers, tablefmt="grid"))


if __name__ == '__main__':
    main()
