# CNT-TEM-Particle-Analysis

Automated TEM image analysis pipeline for extracting nanoparticle size distributions from transmission electron microscopy images of carbon nanotube catalyst particles.

## Overview

This pipeline uses [Detectron2](https://github.com/facebookresearch/detectron2) (Mask R-CNN with ResNet-101-FPN backbone) to detect and segment catalyst nanoparticles in TEM images. It outputs equivalent circular diameters for each detected particle and generates size distribution histograms.

## Method

1. **Training**: A Mask R-CNN model (ResNet-101-FPN, pretrained on COCO) is fine-tuned on manually annotated TEM images of iron catalyst nanoparticles using COCO-format instance segmentation annotations.
2. **Inference**: The trained model segments individual nanoparticles in new TEM images.
3. **Size extraction**: For each detected mask, the equivalent circular diameter is computed as `d = 2 * sqrt(area / pi)` in pixels, then converted to nanometers using a user-specified calibration factor (`pixels_per_nm`).

## Repository Structure

```
├── train_model.py          # Training script
├── analyze_images.py       # Batch inference and size distribution extraction
├── README.md
└── example/                # Example input/output (optional)
```

## Requirements

- Python 3.8+
- PyTorch >= 1.9
- Detectron2 (see [installation instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html))
- OpenCV, NumPy, Matplotlib

## Usage

### Training

Prepare your dataset in COCO instance segmentation format with `train/` and `valid/` directories, each containing images and an `_annotations.coco.json` file.

```python
python train_model.py --dataset_dir /path/to/dataset --output_dir /path/to/output --num_classes 2 --max_iter 1000
```

### Batch Analysis

```python
python analyze_images.py --input_folder /path/to/images --output_folder /path/to/results --model_weights /path/to/model_final.pth --pixels_per_nm 7.5
```

The script outputs annotated images with detected masks and a combined particle size distribution histogram.

## Calibration

The `pixels_per_nm` parameter must be determined from the scale bar in your TEM images. This value depends on the magnification and detector used.

## Citation

If you use this code, please cite:

> Junnarkar, J. et al., "Deep Jet Injection of Plasma Generated Catalyst Aerosol for High Yield Carbon Nanotube Synthesis," *Carbon* (2026).

## License

MIT License
