from data import load_images, load_ground_truth
from roi import detect_disc, extract_roi, segment_disc, segment_cup
from roi import remove_vessels
import cv2
import matplotlib.pyplot as plt

def main():
    images = load_images('refuge_images/retinography/')
    ground_truths = load_ground_truth('refuge_images/ground_truth/')

    for idx, ((filename, image), gt) in enumerate(zip(images, ground_truths)):

        centroid, bounding_box = detect_disc(image)
        roi = extract_roi(image, bounding_box)

        vessels_removed = remove_vessels(roi)

        disc_mask, disc_adjusted = segment_disc(roi, vessels_removed, f"Disc_{idx}")

        cup_mask = segment_cup(roi, disc_adjusted, f"Cup_{idx}")

        fig, axes = plt.subplots(1, 5, figsize=(16, 8))
        fig.suptitle(filename)
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        axes[1].set_title("ROI")
        axes[1].axis('off')

        axes[2].imshow(cv2.cvtColor(vessels_removed, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Vessels Removed")
        axes[2].axis('off')

        axes[3].imshow(disc_mask, cmap='gray')
        axes[3].set_title("Optic Disc")
        axes[3].axis('off')

        axes[4].imshow(cup_mask, cmap='gray')
        axes[4].set_title("Cup")
        axes[4].axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
