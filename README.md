```
title: the Realization of DeepFM in Pytorch
data: 2026/04/17 - 2026/04/19
author: Ale
```

## Description
* Firstly, build the network model based on the paper **"DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"**.
* Secondly, train the model on the **Criteo Display Advertising Challenge** dataset using PyTorch.
* Finally, visualize the training logs, evaluate the model with CTR metrics (like AUC, LogLoss), and save the best checkpoints.

## Preparation
* Environment Requirements:
  - Python 3.x
  - PyTorch
  - pandas, numpy, scikit-learn, etc. (Can be installed via `pip` or `conda`)

* Dataset Download & Processing:
  Download the dataset by running the following script:
  ```shell
  python download.py
  ```
  Alternatively, you can download it directly from the [Kaggle Website](https://www.kaggle.com/datasets/mrkmakr/criteo-dataset?resource=download).

## Usage
1. **Data Processing:** Check `docs/process_raw_data.md` for details on how the raw Criteo dataset is transformed.
2. **Training:** Run the Jupyter Notebook `train.ipynb` for an interactive training process, or execute the training script:
   ```shell
   python scripts/train.py
   ```
3. **Evaluation & Metrics:** See `docs/metric.md` for an in-depth analysis of the metrics used in this CTR scenario (e.g., AUC, LogLoss).
4. **Results:** The best model weights will be saved to the `checkpoints/` directory (e.g., `deepfm_best.pth`), and training plots will be saved in `logger/plots/`.

## Project Structure

```text
trainingDeepFM/
├── checkpoints/              # Saved model weights (.pth)
│   └── deepfm_best.pth
├── data/                     # Datasets
│   ├── Criteo/
│   └── Criteo_small/
├── docs/                     # Documentation and analyses
│   ├── metric.md             # Explanation of CTR metrics
│   ├── process_raw_data.md   # Data preprocessing details
│   └── assets/
├── logger/                   # Training logs and visualizations
│   └── plots/
├── scripts/                  # Core Python source code
│   ├── basic/                # Basic NN components
│   │   ├── activation.py
│   │   ├── features.py
│   │   └── layers.py
│   ├── dataset.py            # DataLoader and Dataset classes
│   ├── model.py              # DeepFM model architecture
│   ├── plot.py               # Plotting utilities
│   ├── train.py              # Training loop
│   └── utils.py              # Helper functions
├── download.py               # Script to download dataset
├── train.ipynb               # Jupyter notebook for training and testing
└── README.md                 # Project entry documentation
```