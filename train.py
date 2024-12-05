import argparse
import logging
import time
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from funcyou.utils import DotDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data import load_image_data
from model import VisionTransformerResNet  # make sure this is the VisionTransformerResNet class from model.py
from utils import set_seed

import wandb

# configure logger
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# vision transformer training function
def train_vision_transformer(
    model,
    dataloader,
    optimizer,
    loss_function,
    num_epochs=50,
    device="cpu",
    data_percent=1.0,
    steps_per_epoch=None,
    save_on_every_n_epochs=5,
    model_path=None,
    xgb_model_path=None,
    xgb_classifier=None,
    checkpoint_path=None,
    start_epoch=0
):
    model.to(device)
    model.train()
    print(f"{model.__class__.__name__} Running on: {device}")
    print(f"Number of epochs: {num_epochs}")
    
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        total_correct_predictions = 0
        total_samples = 0

        epoch_progress = tqdm(
            dataloader, desc=f"Epoch [{epoch + 1:2}/{num_epochs:2}]"
        )

        for batch in epoch_progress:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(images)  # forward pass
            outputs = outputs.mean(dim=1)
            outputs = torch.sigmoid(outputs)
            predictions = torch.round(outputs)
            
            loss = loss_function(outputs, labels.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            epoch_progress.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Accuracy": f"{(total_correct_predictions / total_samples) * 100:.2f}%"
            })

        average_loss = total_loss / len(dataloader)
        average_accuracy = (total_correct_predictions / total_samples) * 100
        print(f"\nEpoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f} - Accuracy: {average_accuracy:.2f}%")
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f} - Accuracy: {average_accuracy:.2f}%")
        
        wandb.log({
            "loss": average_loss,
            "accuracy": average_accuracy,
            "epoch": epoch + 1  # Log the current epoch
        })

        # Save checkpoint every 'save_on_every_n_epochs' epochs
        if (epoch + 1) % save_on_every_n_epochs == 0 and checkpoint_path:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'wandb_run_id': wandb.run.id  # Save the wandb run ID
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        # Save model and xgb model if paths are provided
        if (epoch + 1) % save_on_every_n_epochs == 0 and model_path:
            torch.save(model.state_dict(), model_path)
            # Log the model checkpoint to wandb
            wandb.save(str(model_path))
            if xgb_model_path and xgb_classifier:
                xgb_classifier.save_model(xgb_model_path)
                # Log the xgboost model
                wandb.save(str(xgb_model_path))


# feature extraction function
def extract_features(model, dataloader, device="cpu"):
    model.to(device)
    model.eval()  # eval mode for feature extraction
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            features, _ = model(images)  # forward pass
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
    
    features = torch.cat(all_features).numpy()
    labels = torch.cat(all_labels).numpy()
    return features, labels


# training function for XGBoost
def train_xgboost(features, labels):
    xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xgb_classifier.fit(features, labels)
    return xgb_classifier


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description='Vision Transformer with XGBoost Training and Testing')
    parser.add_argument('image_directory', type=str, help='Path to the directory containing training images')
    parser.add_argument('--test', action='store_true', default=False, help='Flag to test a directory')
    parser.add_argument('--test-directory', type=str, help='Path to the directory containing test images when testing')

    args = parser.parse_args()
    
    config = DotDict.from_toml('config.toml')  # load config    
    set_seed(config.seed)
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths for saving models and checkpoints
    model_path = Path(config.model_path)
    xgb_model_path = model_path.with_suffix('.xgb')
    checkpoint_path = model_path.with_suffix('.ckpt')
    model_path.parent.mkdir(exist_ok=True)
    
    # Initialize wandb with resume functionality
    wandb_run_id = None
    start_epoch = 0

    if checkpoint_path.exists():
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        wandb_run_id = checkpoint.get('wandb_run_id')
        start_epoch = checkpoint['epoch'] + 1
    elif os.path.exists('wandb_run_id.txt'):
        # Load wandb run ID if it exists
        with open('wandb_run_id.txt', 'r') as f:
            wandb_run_id = f.read().strip()
    
    # Initialize wandb
    wandb.init(project="cs229-final", config=config, resume="allow", id=wandb_run_id)
    
    # Save the wandb run ID for future resumes
    with open('wandb_run_id.txt', 'w') as f:
        f.write(wandb.run.id)
    
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(), 
    ])

    # load data
    train_dataloader = load_image_data(args.image_directory, config.image_size, config.batch_size, use_sampler=True, transform=train_transform)
    
    # Initialize model and move to device before creating optimizer
    model = VisionTransformerResNet(config)
    model.to(config.device)
    print("Number of parameters: ", count_parameters(model))
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # initialize xgb classifier
    xgb_classifier = None

    # Load model and optimizer states if checkpoint exists
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Move optimizer state tensors to device
            for state in optimizer.state.values():
                if isinstance(state, dict):
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(config.device)
                elif isinstance(state, torch.Tensor):
                    state = state.to(config.device)

            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
            if xgb_model_path.exists():
                xgb_classifier = xgb.XGBClassifier()
                xgb_classifier.load_model(str(xgb_model_path))
            else:
                print(f"XGBoost model file not found at {xgb_model_path}. Proceeding to train a new model.")
                xgb_classifier = None
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise e
    elif model_path.exists():
        # For backward compatibility if the checkpoint doesn't exist but model weights do
        try:
            # load state dict
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
            if xgb_model_path.exists():
                xgb_classifier = xgb.XGBClassifier()
                xgb_classifier.load_model(str(xgb_model_path))
            else:
                print(f"XGBoost model file not found at {xgb_model_path}. Proceeding to train a new model.")
                xgb_classifier = None
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    else:
        print(f"Model file not found at {model_path}. Training from scratch.")

    # Watch the model with wandb
    wandb.watch(model, log='all')
    
    # train ViT
    print("training vision transformer...")
    train_vision_transformer(
        model,
        train_dataloader,
        optimizer,
        loss_function,
        num_epochs=config.num_epochs,
        device=config.device,
        model_path=model_path,
        xgb_model_path=xgb_model_path,
        xgb_classifier=xgb_classifier,
        checkpoint_path=checkpoint_path,
        start_epoch=start_epoch
    )
    
    # extract features from ViT
    print("extracting features from vision transformer...")
    features, labels = extract_features(model, train_dataloader, device=config.device)
    
    # train XGBoost classifier on extracted features
    print("training XGBoost classifier...")
    xgb_classifier = train_xgboost(features, labels)
    xgb_classifier.save_model(str(xgb_model_path))
    print(f"XGBoost classifier saved at {xgb_model_path}")

    # evaluate on the test dataset if provided
    if args.test:
        logger.info(f'testing data: {args.test_directory}')
        test_dataloader = load_image_data(args.test_directory, config.image_size, config.batch_size)

        # extract features from test set
        print("extracting test features...")
        test_features, test_labels = extract_features(model, test_dataloader, device=config.device)

        # predict using XGBoost classifier
        y_pred = xgb_classifier.predict(test_features)

        # calculate evaluation metrics
        accuracy = accuracy_score(test_labels, y_pred)
        precision = precision_score(test_labels, y_pred)
        recall = recall_score(test_labels, y_pred)
        f1 = f1_score(test_labels, y_pred)

        # log and print metrics
        logger.info(f'Accuracy: {accuracy}')
        logger.info(f'Precision: {precision}')
        logger.info(f'Recall: {recall}')
        logger.info(f'F1 Score: {f1}')
        logger.info('-'*40)

        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')

        # Log evaluation metrics to wandb
        wandb.log({'test_accuracy': accuracy, 'test_precision': precision, 'test_recall': recall, 'test_f1': f1})

    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
