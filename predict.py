import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from funcyou.utils import DotDict
from PIL import Image
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from data import load_image_data
from model import VisionTransformer
from utils import predict_model, set_seed

# Configure logger
logging.basicConfig(filename='prediction.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()


def load_and_preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image

def predict_directory(directory_path, model, transform, batch_size=8, device='cpu'):
    model.eval()

    
    dataset = ImageFolder(directory_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    image_paths = [i[0] for i in dataset.imgs]

    predictions = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Predicting"):
            images = images.to(device)
            outputs,_ = model(images)
            predicted = torch.round(outputs.squeeze())
            
            predictions.extend(predicted.cpu().numpy())

    print('image paths: ', len(image_paths))
    print('pred shap: ', len(predictions))
    return pd.DataFrame(
        {
            "Image_Path": image_paths,
            "Predicted_Class": predictions,
        }
    )

def test_directory(directory_path, model, transform, batch_size=8, device='cpu'):
    model.eval()
    directory_path = Path(directory_path)

    dataset = ImageFolder(directory_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Predicting"):
            images = images.to(device)
            labels = labels.to(device)

            outputs,_ = model(images)
            predicted = torch.round(outputs.squeeze())
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    return pd.DataFrame(
        {
            "Image_Path": dataset.imgs,
            "True_Class": y_true,
            "Predicted_Class": y_pred,
        }
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pneumonia Detection using Vision Transformer')
    parser.add_argument('image_directory', type=str, help='Path to the directory containing test folder that contains images')
    parser.add_argument('--test', action='store_true', default=False, help='flags to test a folder')

    args = parser.parse_args()

    config = DotDict.from_toml('config.toml')
    set_seed(config.seed)

    model = VisionTransformer(config)
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the trained model weights
    model.load_state_dict(torch.load(config.xgb_model_path))
    model.to(config.device)
    model.eval()
    
    # Define transformations for test images (should be the same as used during training)
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.RandomHorizontalFlip(),
        ransforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(), 
    ])
    
    image_directory = Path(args.image_directory) 
    test = args.test
    
    logger.info(f'Testing directory: {image_directory}')
    
    if test:
        result_df = test_directory(image_directory, model, transform, batch_size=8, device=config.device)
        
        # Calculate evaluation metrics
        y_true = result_df["True_Class"].values
        y_pred = result_df["Predicted_Class"].values
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Log the metrics
        logger.info(f'Accuracy: {accuracy}')
        logger.info(f'Precision: {precision}')
        logger.info(f'Recall: {recall}')
        logger.info(f'F1 Score: {f1}')
        
        # Print metrics in the terminal
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
    else:
        result_df = predict_directory(image_directory, model, transform)
    
    # Log the number of images
    num_images = len(result_df)
    logger.info(f'Number of images: {num_images}')
    
    ttime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    ttime = str(ttime).replace('/','-')
    filename = f'output-{ttime}.csv'
    print(result_df)  # Display the first few rows of the DataFrame
    result_df.to_csv(filename)
    print(f'Output saved in {filename}!')
    logger.info(f'Output saved in {filename}!')
    logger.info('-'*40)
