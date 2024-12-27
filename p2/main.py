from data import load_images, load_ground_truth
from roi import get_roi, get_roi_from_disc, extract_roi, segment_disc, segment_cup, remove_vessels_disc, remove_vessels_cup
from metrics import compute_iou, compute_dice
import cv2
import os
import matplotlib.pyplot as plt

def main():
    images = load_images('refuge_images/retinography/')
    ground_truths = load_ground_truth('refuge_images/ground_truth/')

    for idx, ((filename, image)) in enumerate(images):
                # Obtener los ground truths correspondientes
        gt_filenameDisc = f"{filename.split('.')[0]}_disc.png"
        gt_filenameCup = f"{filename.split('.')[0]}_cup.png"
        gt_disc = ground_truths['disc'].get(gt_filenameDisc, None)
        gt_cup = ground_truths['cup'].get(gt_filenameCup, None)


        roi_coordinates = get_roi(image, 200)

        roi_disc = extract_roi(image, roi_coordinates)

        vessels_removed_disc = remove_vessels_disc(roi_disc, 190, 8)

        disc_mask = segment_disc(image, roi_coordinates, vessels_removed_disc, f"Disc_{idx}")

        small_roi_coordinates = get_roi_from_disc(image, disc_mask, 130)

        roi_disc_mask = extract_roi(disc_mask, small_roi_coordinates)

        roi_cup = extract_roi(image, small_roi_coordinates)

        vessels_removed_cup = remove_vessels_cup(roi_cup, roi_disc_mask, 180, 185, 2) # 160 200

        cup_mask = segment_cup(image, small_roi_coordinates, vessels_removed_cup, f"Cup_{idx}")

        # Crear carpetas si no existen
        os.makedirs('result/disc', exist_ok=True)
        os.makedirs('result/cup', exist_ok=True)

        # Guardar disc y cup con el nombre de la imagen en las carpetas result/disc y result/cup respectivamente
        disc_filename = f"result/disc/{filename.split('.')[0]}_disc.png"
        cup_filename = f"result/cup/{filename.split('.')[0]}_cup.png"
        cv2.imwrite(disc_filename, disc_mask)
        cv2.imwrite(cup_filename, cup_mask)
        
        # Calcular métricas
        if gt_disc is not None:
            iou_disc = compute_iou(disc_mask, gt_disc)
            dice_disc = compute_dice(disc_mask, gt_disc)
            print(f"Image: {filename}")
            print(f"  Optic Disc - IoU: {iou_disc:.4f}, Dice: {dice_disc:.4f}")
        else:
            print(f"Image: {filename}")
            print(f"  No ground truth found for optic disc")

        if gt_cup is not None:
            iou_cup = compute_iou(cup_mask, gt_cup)
            dice_cup = compute_dice(cup_mask, gt_cup)
            print(f"  Cup - IoU: {iou_cup:.4f}, Dice: {dice_cup:.4f}")
        else:
            print(f"  No ground truth found for cup")

        # Visualización
        fig, axes = plt.subplots(1, 7, figsize=(16, 8))
        fig.suptitle(filename)
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(roi_disc, cv2.COLOR_BGR2RGB))
        axes[1].set_title("roi_image")
        axes[1].axis('off')

        axes[2].imshow(cv2.cvtColor(vessels_removed_disc, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Vessels Removed")
        axes[2].axis('off')

        axes[3].imshow(disc_mask, cmap='gray')
        axes[3].set_title("Optic Disc")
        axes[3].axis('off')

        axes[4].imshow(cv2.cvtColor(roi_cup, cv2.COLOR_BGR2RGB))
        axes[4].set_title("roi_image")
        axes[4].axis('off')

        axes[5].imshow(cv2.cvtColor(vessels_removed_cup, cv2.COLOR_BGR2RGB))
        axes[5].set_title("Vessels Removed")
        axes[5].axis('off')

        axes[6].imshow(cup_mask, cmap='gray')
        axes[6].set_title("Cup")
        axes[6].axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
