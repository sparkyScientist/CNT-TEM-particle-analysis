"""
analyze_images.py

Runs a trained Mask R-CNN model on a folder of TEM images to detect
nanoparticles, extract equivalent circular diameters, and produce
a combined particle size distribution.

Usage:
    python analyze_images.py \
        --input_folder /path/to/images \
        --output_folder /path/to/results \
        --model_weights /path/to/model_final.pth \
        --pixels_per_nm 7.5 \
        --num_classes 2 \
        --confidence 0.5
"""

import argparse
import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

setup_logger()

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch nanoparticle detection and size analysis from TEM images"
    )
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Folder containing TEM images to analyze")
    parser.add_argument("--output_folder", type=str, default="./results",
                        help="Folder to save annotated images and results")
    parser.add_argument("--model_weights", type=str, required=True,
                        help="Path to trained model weights (model_final.pth)")
    parser.add_argument("--pixels_per_nm", type=float, required=True,
                        help="Calibration factor: pixels per nanometer")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of object classes")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Detection confidence threshold (0-1)")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Dataset directory for metadata (optional)")
    parser.add_argument("--histogram_bins", type=int, default=25,
                        help="Number of bins for the size distribution histogram")
    return parser.parse_args()


def setup_predictor(args):
    """Configure and return a Detectron2 predictor."""
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.MODEL.WEIGHTS = args.model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence

    # Register dataset for metadata if provided
    if args.dataset_dir is not None:
        train_json = os.path.join(args.dataset_dir, "train", "_annotations.coco.json")
        train_imgs = os.path.join(args.dataset_dir, "train")
        try:
            DatasetCatalog.remove("nanoparticles_train")
        except KeyError:
            pass
        register_coco_instances("nanoparticles_train", {}, train_json, train_imgs)
        cfg.DATASETS.TRAIN = ("nanoparticles_train",)

    return DefaultPredictor(cfg), cfg


def extract_diameters(masks, pixels_per_nm):
    """
    Compute equivalent circular diameters from binary masks.
    
    For each mask, the area in pixels is computed and converted
    to an equivalent circular diameter: d = 2 * sqrt(area / pi).
    The result is then scaled to nanometers using the calibration factor.
    
    Parameters
    ----------
    masks : np.ndarray
        Array of binary masks, shape (N, H, W).
    pixels_per_nm : float
        Calibration factor (pixels per nanometer).
    
    Returns
    -------
    diameters_nm : list of float
        Equivalent circular diameters in nanometers.
    """
    diameters_nm = []
    for mask in masks:
        area_pixels = np.sum(mask)
        diameter_pixels = 2 * math.sqrt(area_pixels / math.pi)
        diameters_nm.append(diameter_pixels / pixels_per_nm)
    return diameters_nm


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    predictor, cfg = setup_predictor(args)

    all_diameters_nm = []
    image_files = sorted([
        f for f in os.listdir(args.input_folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])

    print(f"Found {len(image_files)} images in '{args.input_folder}'")

    for filename in image_files:
        image_path = os.path.join(args.input_folder, filename)
        im = cv2.imread(image_path)

        if im is None:
            print(f"  Warning: could not read {filename}, skipping.")
            continue

        outputs = predictor(im)
        instances = outputs["instances"].to("cpu")

        # Extract diameters
        masks = instances.pred_masks.numpy()
        diameters = extract_diameters(masks, args.pixels_per_nm)
        all_diameters_nm.extend(diameters)

        # Save annotated image
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]) if cfg.DATASETS.TRAIN else {}
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.8)
        out = v.draw_instance_predictions(instances)
        output_path = os.path.join(args.output_folder, filename)
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

        print(f"  {filename}: {len(instances)} particles detected")

    # Save diameter data
    data_path = os.path.join(args.output_folder, "diameters_nm.csv")
    np.savetxt(data_path, all_diameters_nm, delimiter=",",
               header="equivalent_diameter_nm", comments="")
    print(f"\nDiameter data saved to {data_path}")

    # Plot histogram
    if all_diameters_nm:
        plt.figure(figsize=(10, 6))
        plt.hist(all_diameters_nm, bins=args.histogram_bins, edgecolor='black')
        plt.xlabel('Equivalent Diameter (nm)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Nanoparticle Size Distribution from TEM', fontsize=14)
        hist_path = os.path.join(args.output_folder, "size_distribution.png")
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Histogram saved to {hist_path}")

        print(f"\nSummary:")
        print(f"  Total particles detected: {len(all_diameters_nm)}")
        print(f"  Mean diameter: {np.mean(all_diameters_nm):.1f} nm")
        print(f"  Median diameter: {np.median(all_diameters_nm):.1f} nm")
        print(f"  Std deviation: {np.std(all_diameters_nm):.1f} nm")
    else:
        print("No particles detected in any image.")


if __name__ == "__main__":
    main()
