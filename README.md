# ViTXGB: Vision Transformer with XGBoost

A PyTorch implementation of Vision Transformer (ViT) with XGBoost classifier for feature extraction and classification.

## Setup

1. Ensure you have Python 3.11 installed
2. Clone the repository:
```bash
git clone https://github.com/SohamGovande/cs229-finalproj.git
```
3. Install the package in editable mode:
```bash
pip install -e .
```

## Training

Launch the training script using:
```bash
python3 -m vit.scripts.train [image_directory] [options]
```

### Arguments

- `image_directory`: Path to directory containing training images. Should be organized in subdirectories by class.
- `--test`: (Optional) Flag to evaluate model on test set
- `--test-directory`: Path to test images directory when using `--test` flag

### Example Usage

Train on training data:
```bash
python3 -m vit.scripts.train data/train
```

Train and evaluate on test set:
```bash
python3 -m vit.scripts.train data/train --test --test-directory data/test
```

## Features

- Vision Transformer (ViT) model with ResNet backbone
- XGBoost classifier on extracted features
- Weights & Biases integration for experiment tracking
- Checkpoint saving and resuming
- Data augmentation with random rotations, flips, perspective and color jittering
- Weighted random sampling to handle class imbalance
- Comprehensive logging and metrics tracking

## Model Outputs

The training script saves several files:

- `{model_path}`: Vision Transformer model weights
- `{model_path}.xgb`: Trained XGBoost classifier
- `{model_path}.ckpt`: Training checkpoint for resuming
- `wandb_run_id.txt`: Weights & Biases run ID for experiment tracking
- `training.log`: Detailed training logs

## Configuration

Model and training parameters can be configured in `config.toml`. Key parameters include:

- `image_size`: Input image dimensions
- `batch_size`: Training batch size
- `learning_rate`: Optimizer learning rate
- `num_epochs`: Number of training epochs
- `seed`: Random seed for reproducibility
- `model_path`: Path to save model weights
