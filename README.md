# Semi-Supervised Image Segmentation with Mean-Teacher and U-Net

This project implements a semi-supervised learning framework for binary image segmentation. The goal is to accurately segment pets (cats and dogs) from the background using a **U-Net** architecture, even when only a small fraction of the training data is labeled.

## The Problem

Training deep learning models for semantic segmentation typically requires large, pixel-perfect labeled datasets (masks). Creating these datasets is extremely costly and time-consuming. This project explores a semi-supervised approach to overcome this challenge by leveraging a large amount of *unlabeled* data to improve model performance.

## Methodology

### 1. Model: U-Net
The core architecture is a **U-Net**, a convolutional neural network (CNN) specifically designed for fast and precise image segmentation. It consists of a contracting (encoder) path to capture context and a symmetric expanding (decoder) path that enables precise localization.

The full PyTorch implementation of the model is in `model_UNet.py`.

### 2. Semi-Supervised Technique: Mean-Teacher
The project uses the **Mean-Teacher** model for consistency regularization. This framework involves two networks:
* **Student Network:** This is the primary model being trained. It learns from both labeled data (using a standard supervised loss) and unlabeled data (using an unsupervised consistency loss).
* **Teacher Network:** This model is not trained via backpropagation. Instead, its weights are an **exponential moving average** of the student network's weights.

The student model is trained to produce consistent predictions on augmented versions of an image, and the "teacher" network provides a more stable, reliable prediction target for the unlabeled data. The training logic can be found in `Train_Mean_Teacher.ipynb`.

### 3. Dataset: Oxford-IIIT Pet
The model is trained on the **Oxford-IIIT Pet dataset**, which contains images of 37 different pet breeds. For this project, all breeds are grouped into a single "pet" class for binary segmentation (pet vs. background).

The `data_into_loaders.py` script handles:
* Downloading the dataset from its source.
* Preprocessing the images and masks.
* Creating PyTorch DataLoaders that provide mixed batches of labeled and unlabeled data, controlled by a `supervised_pct` parameter.

### 4. Loss Functions & Augmentation
* **Supervised Loss:** The model uses **Dice Loss** on the labeled data, which is well-suited for segmentation tasks, especially with class imbalance (e.g., more background pixels than pet pixels).
* **Unsupervised Loss:** A consistency loss (also based on Dice Loss) is calculated between the student's predictions and the teacher's (pseudo-label) predictions on unlabeled data.
* **Data Augmentation:** The script `data_augmentation.py` implements several random augmentations (e.g., Gaussian Noise, Color Jitter, Saturation, Invert) to improve model robustness and for the consistency training.

## Key Findings

The report compares the performance of the semi-supervised Mean-Teacher (MT) models against fully-supervised "Lower Bound" (LB) models trained on the same small percentage of labeled data.

| Model | Labeled Data | Accuracy (%) | IoU (%) |
| :--- | :--- | :--- | :--- |
| **Upper Bound** (Full Supervision) | 100% | 90.29 | 79.23 |
| **M25 L** (Lower Bound) | 25% | 86.93 | 73.01 |
| **M25** (Mean-Teacher) | 25% | **88.24** | **75.46** |
| **M10 L** (Lower Bound) | 10% | 85.03 | 70.52 |
| **M10** (Mean-Teacher) | 10% | **85.49** | 69.97 |
| **M05 L** (Lower Bound) | 5% | 83.04 | 67.83 |
| **M05** (Mean-Teacher) | 5% | **83.23** | **68.47** |

The results show that the Mean-Teacher models consistently improved performance over the baseline models, demonstrating the value of using unlabeled data. The M25 model, for example, recovered a significant portion of the performance lost from using 75% less labeled data.

## Files in this Repository

* `report.pdf`: The full academic report detailing the methodology, experiments, and results.
* `Train_Mean_Teacher.ipynb`: Jupyter Notebook to train the semi-supervised Mean-Teacher model.
* `Train_Supervised.ipynb`: Jupyter Notebook to train the supervised baseline (Upper and Lower Bound) models.
* `model_UNet.py`: PyTorch implementation of the U-Net architecture.
* `data_into_loaders.py`: Scripts to download, process, and load the Oxford-IIIT Pet dataset.
* `data_augmentation.py`: Contains all image augmentation functions.
* `utils.py`: Utility functions for training, including loss functions and model evaluation.

## How to Use

1.  Ensure you have PyTorch, torchvision, and other standard libraries (numpy, PIL) installed.
2.  To download the data, call the `download_data()` function from `data_into_loaders.py` (this is also included in the training notebooks).
3.  To train a **semi-supervised model**, configure the `supervised_pct` variable in `Train_Mean_Teacher.ipynb` and run the notebook.
4.  To train a **fully supervised baseline**, configure the `pct_data` variable in `Train_Supervised.ipynb` and run the notebook.
