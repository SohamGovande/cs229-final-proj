# Vision Transformer (ViT)
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from tqdm import tqdm


def predict_model(model, dataloader, device="cpu"):
    model.to(device)

    y_true = []
    y_pred = []

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs, _ = model(images)
            outputs = outputs.squeeze()
            predictions = torch.round(outputs)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())


    return y_true, y_pred



def validate_model(
    model,
    dataloader,
    loss_function,
    device="cpu",
):
    model.to(device)
    print(f"Validating {model.__class__.__name__} on: {device}")

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.inference_mode():
        validation_progress = tqdm(
            dataloader, desc="Validation"
        )

        for batch in validation_progress:
            image, label = batch
            image = image.to(device)
            label = label.to(device)

            outputs, _ = model(image)
            outputs = outputs.squeeze()
            predictions = torch.round(outputs)

            loss = loss_function(outputs, label.to(outputs.dtype))
            
            total_loss += loss.item()
            correct_predictions += (predictions.to(torch.int32) == label.to(torch.int32)).sum().item()
            total_samples += label.size()[0]

            formatted_loss = f"{loss.item():.8f}"
            accuracy = (correct_predictions / total_samples) * 100
            formatted_accuracy = f"{accuracy:.2f}%"

            validation_progress.set_postfix(
                {"Loss": formatted_loss, "Accuracy": formatted_accuracy}
            )

    average_loss = total_loss / len(dataloader)
    accuracy = (correct_predictions / (total_samples + 1e-7)) * 100

    print(f"Validation - Average Loss: {average_loss:.8f} - Accuracy: {accuracy:.2f}%")
    print()




def calculate_class_weights_from_directory(directory_path:Path):
    """
    This function calculates the class weights for a given directory of images.

    Args:
    directory_path (str): The path to the directory of images.

    Returns:
    (list, list, np.ndarray): The class distribution, class weights, and bincount.
    """

    # Create a Path object for the directory
    directory = Path(directory_path)

    # Get a list of subdirectories (classes)
    classes = sorted([d.name for d in directory.iterdir() if d.is_dir()])

    # Initialize class_dist to store class labels
    class_dist = []
    
    # Iterate through each class directory
    for idx, class_name in enumerate(classes):
        class_path = directory / class_name
        num_files = len(list(class_path.glob('*')))
        class_dist.extend([idx] * num_files)

    counter = Counter(class_dist)
    # Calculate the class counts using bincount
    bincount = np.bincount(class_dist)

    # Calculate class weights as 1 / bincount for each class
    class_weights = 1.0 / bincount[class_dist]
    

    return class_dist, class_weights, counter



def set_seed(seed_value=42):
    """
    Set seed for reproducibility in PyTorch, NumPy, and Python's random library.

    Args:
        seed_value (int): The seed value to set.

    Returns:
        None
    """
    # Set seed for NumPy
    np.random.seed(seed_value)

    # Set seed for Python's random library
    random.seed(seed_value)

    # Set seed for PyTorch
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set seed for CuDNN (if available)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set the environment variable for better reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed_value)

def extract_features(model, dataloader, device='cpu'):
    """
    Extract features using a given model.
    Args:
        model: The Vision Transformer or other model for feature extraction.
        dataloader: DataLoader for the dataset.
        device: Device to run the model on.
    Returns:
        features: Extracted features for each input in the dataloader.
        labels: Corresponding labels for each input.
    """
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    all_features = []
    all_labels = []

    with torch.no_grad():  # No need to track gradients during inference
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            # Perform a forward pass and get features
            features, _ = model(images)  # assuming model returns (features, attention_weights)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all features and labels into a single tensor
    features = torch.cat(all_features).numpy()
    labels = torch.cat(all_labels).numpy()
    return features, labels
