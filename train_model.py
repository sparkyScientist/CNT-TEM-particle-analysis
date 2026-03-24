"""
train_model.py

Fine-tunes a Mask R-CNN (ResNet-101-FPN) model on TEM nanoparticle images
using Detectron2. Expects COCO-format annotations.

Usage:
    python train_model.py \
        --dataset_dir /path/to/dataset \
        --output_dir /path/to/output \
        --num_classes 2 \
        --max_iter 1000 \
        --lr 0.00025
"""

import argparse
import os
import torch

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances

setup_logger()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN for nanoparticle instance segmentation"
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Root directory containing train/ and valid/ subdirectories")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save model weights and logs")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of object classes in the dataset")
    parser.add_argument("--max_iter", type=int, default=1000,
                        help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=0.00025,
                        help="Base learning rate")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Images per batch")
    return parser.parse_args()


def main():
    args = parse_args()

    # Register datasets
    train_json = os.path.join(args.dataset_dir, "train", "_annotations.coco.json")
    train_imgs = os.path.join(args.dataset_dir, "train")
    val_json = os.path.join(args.dataset_dir, "valid", "_annotations.coco.json")
    val_imgs = os.path.join(args.dataset_dir, "valid")

    register_coco_instances("nanoparticles_train", {}, train_json, train_imgs)
    register_coco_instances("nanoparticles_val", {}, val_json, val_imgs)

    # Configure model
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    )
    cfg.DATASETS.TRAIN = ("nanoparticles_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    )
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes

    os.makedirs(args.output_dir, exist_ok=True)
    cfg.OUTPUT_DIR = args.output_dir

    # Train
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    print(f"Starting training for {args.max_iter} iterations...")
    trainer.train()
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
