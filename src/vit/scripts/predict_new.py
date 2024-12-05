import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np  
import pandas as pd
import torch
from funcyou.utils import DotDict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,          
    average_precision_score 
)
from torchvision import transforms
from tqdm import tqdm

from data import load_image_data
from model import DenseNet121Baseline, VisionTransformerResNet  
from src.vit.utils import set_seed, extract_features

# Configure logger to print to terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def main():
    try:
        parser = argparse.ArgumentParser(description='Model Prediction')
        parser.add_argument('image_directory', type=str, help='Path to the directory containing test images')
        parser.add_argument('--model', type=str, default='vit', choices=['densenet', 'vit'], help='Model to evaluate')

        args = parser.parse_args()

        # Load configuration
        config = DotDict.from_toml('config.toml')
        set_seed(config.seed)
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if args.model == 'densenet':
            # Initialize model
            model = DenseNet121Baseline(num_classes=config.num_classes)
            model.load_state_dict(torch.load(config.densenet_model_path))
            model.to(config.device)
            model.eval()

            # Define transformations for test images
            transform = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(), 
            ])

            # Load images with labels
            image_directory = Path(args.image_directory)
            test_dataloader = load_image_data(
                str(image_directory),
                config.image_size,
                config.batch_size,
                transform=transform
            )
            logger.info(f'Running predictions on directory: {image_directory}')

            y_true = []
            y_pred = []
            y_scores = []  # Added to store predicted probabilities

            with torch.no_grad():
                for images, labels in tqdm(test_dataloader, desc="Evaluating"):
                    images = images.to(config.device)
                    labels = labels.to(config.device)
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)  # Get probabilities
                    _, predicted = torch.max(outputs, 1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
                    y_scores.extend(probs.cpu().numpy())  # Store probabilities

            y_scores = np.vstack(y_scores)  

            # Calculate and print metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='binary', pos_label=1)
            recall = recall_score(y_true, y_pred, average='binary', pos_label=1)
            f1 = f1_score(y_true, y_pred, average='binary', pos_label=1)
            conf_matrix = confusion_matrix(y_true, y_pred)

            # Compute ROC AUC and Average Precision Score
            if config.num_classes == 2:
                roc_auc = roc_auc_score(y_true, y_scores[:, 1])
                pr_auc = average_precision_score(y_true, y_scores[:, 1])
            else:
                roc_auc = roc_auc_score(y_true, y_scores, multi_class='ovr')
                pr_auc = average_precision_score(y_true, y_scores)

            print(f'Accuracy: {accuracy:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1 Score: {f1:.4f}')
            print(f'ROC AUC Score: {roc_auc:.4f}')  
            print(f'Average Precision Score (AUPRC): {pr_auc:.4f}') 
            print('Confusion Matrix:')
            print(conf_matrix)

            # Save predictions to a CSV file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"densenet_predictions_{timestamp}.csv"
            image_paths = [img_path for img_path, _ in test_dataloader.dataset.imgs]
            predictions_df = pd.DataFrame({
                "Image_Path": image_paths,
                "True_Class": y_true,
                "Predicted_Class": y_pred
            })
            predictions_df.to_csv(output_filename, index=False)
            print(f"Predictions saved to {output_filename}")

        elif args.model == 'vit':
            import xgboost as xgb  # Import xgboost here
            # Initialize model
            model = VisionTransformerResNet(config)
            model.load_state_dict(torch.load(config.model_path))
            model.to(config.device)
            model.eval()

            # Load XGBoost classifier
            xgb_classifier = xgb.XGBClassifier()
            xgb_classifier.load_model(config.large_xgb_model_path)

            # Define transformations for test images
            transform = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(), 
            ])

            # Load images with labels
            image_directory = Path(args.image_directory)
            test_dataloader = load_image_data(
                str(image_directory),
                config.image_size,
                config.batch_size,
                transform=transform
            )
            logger.info(f'Running predictions on directory: {image_directory}')

            # Extract features for XGBoost and get true labels
            features, y_true = extract_features(model, test_dataloader, device=config.device)

            print(f"Extracted feature shape: {features.shape}")  # Should show (number_of_samples, feature_dim)
            
            # Make predictions with XGBoost
            predictions = xgb_classifier.predict(features)
            probs = xgb_classifier.predict_proba(features)  # Added to get probabilities

            # Calculate and print metrics
            accuracy = accuracy_score(y_true, predictions)
            precision = precision_score(y_true, predictions, average='binary', pos_label=1)
            recall = recall_score(y_true, predictions, average='binary', pos_label=1)
            f1 = f1_score(y_true, predictions, average='binary', pos_label=1)
            conf_matrix = confusion_matrix(y_true, predictions)

            # Compute ROC AUC and Average Precision Score
            if config.num_classes == 2:
                roc_auc = roc_auc_score(y_true, probs[:, 1])
                pr_auc = average_precision_score(y_true, probs[:, 1])
            else:
                roc_auc = roc_auc_score(y_true, probs, multi_class='ovr')
                pr_auc = average_precision_score(y_true, probs)

            print(f'Accuracy: {accuracy:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1 Score: {f1:.4f}')
            print(f'ROC AUC Score: {roc_auc:.4f}')  
            print(f'Average Precision Score (AUPRC): {pr_auc:.4f}')  
            print('Confusion Matrix:')
            print(conf_matrix)

            # Save predictions to a CSV file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"vit_predictions_{timestamp}.csv"
            image_paths = [img_path for img_path, _ in test_dataloader.dataset.imgs]
            predictions_df = pd.DataFrame({
                "Image_Path": image_paths,
                "True_Class": y_true,
                "Predicted_Class": predictions
            })
            predictions_df.to_csv(output_filename, index=False)
            print(f"Predictions saved to {output_filename}")

        else:
            print(f"Unsupported model type: {args.model}")
            exit(1)
    
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
