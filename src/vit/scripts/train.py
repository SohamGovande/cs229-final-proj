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
from torchvision import transforms
from tqdm import tqdm

from ..data import load_image_data
from ..model import ViT
from ..utils import set_seed

import wandb

# Configure logger
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger()


def count_parameters(model):
    """Count the total number of trainable parameters in the given model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_xgboost(features, labels):
    """
    Train an XGBoost classifier on extracted features and labels.

    Args:
        features (np.ndarray): Array of extracted features from the model.
        labels (np.ndarray): Array of corresponding labels.

    Returns:
        xgb.XGBClassifier: Trained XGBoost classifier.
    """
    xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xgb_classifier.fit(features, labels)
    return xgb_classifier


def extract_features(model, dataloader, device="cpu"):
    """
    Extract features from a given model using a provided dataloader.

    Args:
        model (nn.Module): Trained model to use for feature extraction.
        dataloader (DataLoader): Dataloader for the dataset from which to extract features.
        device (str): Device to use for computation.

    Returns:
        tuple: (features, labels) as numpy arrays.
    """
    model.to(device)
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            # Forward pass through the model
            features, _ = model(images)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    features = torch.cat(all_features).numpy()
    labels = torch.cat(all_labels).numpy()
    return features, labels


def train_vision_transformer(
    model,
    dataloader,
    optimizer,
    loss_function,
    total_epochs=50,
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
    """
    Train a Vision Transformer model.

    Args:
        model (nn.Module): Vision Transformer model instance.
        dataloader (DataLoader): Dataloader providing training data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_function (nn.Module): Loss function.
        total_epochs (int): Total number of epochs to train.
        device (str): Device for training computation.
        data_percent (float): Not used directly (but must remain to preserve behavior).
        steps_per_epoch (int): Not used directly (but must remain to preserve behavior).
        save_on_every_n_epochs (int): Frequency of saving model checkpoints.
        model_path (Path): Path to save the model weights.
        xgb_model_path (Path): Path to save the XGBoost model.
        xgb_classifier (xgb.XGBClassifier): XGBoost classifier instance, if any.
        checkpoint_path (Path): Path to save PyTorch checkpoints.
        start_epoch (int): Epoch number to start/resume from.

    Returns:
        None
    """
    model.to(device)
    model.train()
    print(f"{model.__class__.__name__} Running on: {device}")
    print(f"Total number of epochs to train: {total_epochs}")

    try:
        for epoch in range(start_epoch, total_epochs):
            total_loss = 0.0
            total_correct_predictions = 0
            total_samples = 0

            epoch_progress = tqdm(
                dataloader, desc=f"Epoch [{epoch + 1}/{total_epochs}]"
            )

            for batch in epoch_progress:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs, _ = model(images)  # Forward pass
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
            print(f"\nEpoch [{epoch + 1}/{total_epochs}] - Loss: {average_loss:.4f} - Accuracy: {average_accuracy:.2f}%")
            logger.info(f"Epoch [{epoch + 1}/{total_epochs}] - Loss: {average_loss:.4f} - Accuracy: {average_accuracy:.2f}%")
            
            # Log to wandb
            wandb.log({
                "loss": average_loss,
                "accuracy": average_accuracy,
                "epoch": epoch + 1
            })

            # Save checkpoint
            if checkpoint_path:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'wandb_run_id': wandb.run.id
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

            # Save model and XGBoost
            should_save = ((epoch + 1) % save_on_every_n_epochs == 0 or (epoch + 1) == total_epochs)
            if should_save and model_path:
                torch.save(model.state_dict(), model_path)
                wandb.save(str(model_path))
                if xgb_model_path and xgb_classifier:
                    xgb_classifier.save_model(xgb_model_path)
                    wandb.save(str(xgb_model_path))

    except KeyboardInterrupt:
        # Handle interruption by saving a checkpoint
        print("Training interrupted. Saving checkpoint...")
        if checkpoint_path:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'wandb_run_id': wandb.run.id
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
        raise

    # Save final checkpoint
    if checkpoint_path:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'wandb_run_id': wandb.run.id
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Final checkpoint saved at {checkpoint_path}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Vision Transformer with XGBoost Training and Testing')
    parser.add_argument('image_directory', type=str, help='Path to the directory containing training images')
    parser.add_argument('--test', action='store_true', default=False, help='Flag to test a directory')
    parser.add_argument('--test-directory', type=str, help='Path to the directory containing test images when testing')
    return parser.parse_args()


def load_config_and_setup_device():
    """Load configuration from 'config.toml' and set device."""
    config = DotDict.from_toml('config.toml')
    set_seed(config.seed)
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return config


def setup_wandb(config, checkpoint_path):
    """
    Initialize wandb, possibly resuming from a previous run, and return run ID and start epoch.
    
    Args:
        config (DotDict): Configuration dictionary.
        checkpoint_path (Path): Path to checkpoint file.

    Returns:
        tuple: (wandb_run_id, start_epoch)
    """
    wandb_run_id = None
    start_epoch = 0

    # Attempt to load from checkpoint if exists
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        wandb_run_id = checkpoint.get('wandb_run_id')
        start_epoch = checkpoint['epoch'] + 1
    elif os.path.exists('wandb_run_id.txt'):
        with open('wandb_run_id.txt', 'r') as f:
            wandb_run_id = f.read().strip()

    # Initialize wandb
    wandb.init(project="cs229-final", config=config, resume="allow", id=wandb_run_id)

    # Save wandb run ID for future resume
    with open('wandb_run_id.txt', 'w') as f:
        f.write(wandb.run.id)

    return wandb_run_id, start_epoch


def initialize_model_and_optimizer(config, model_path, xgb_model_path):
    """
    Initialize the ViT model and optimizer. If a saved model or XGBoost model is found, load it.

    Args:
        config (DotDict): Configuration dictionary.
        model_path (Path): Path to saved model weights.
        xgb_model_path (Path): Path to saved XGBoost model.

    Returns:
        tuple: (model, optimizer, xgb_classifier)
    """
    model = ViT(config)
    model.to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    xgb_classifier = None

    # Try loading existing model weights (if any)
    if model_path.exists():
        try:
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
            if xgb_model_path.exists():
                xgb_classifier = xgb.XGBClassifier()
                xgb_classifier.load_model(str(xgb_model_path))
            else:
                print(f"XGBoost model file not found at {xgb_model_path}. Proceeding to train a new model.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    else:
        print(f"Model file not found at {model_path}. Training from scratch.")

    return model, optimizer, xgb_classifier


def resume_training_if_checkpoint_exists(model, optimizer, xgb_classifier, checkpoint_path, xgb_model_path, start_epoch, config):
    """
    If a checkpoint exists, load state dictionaries and possibly resume training.
    Also handle the optimizer states and load XGBoost classifier if available.

    Args:
        model (nn.Module): Model instance.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        xgb_classifier (xgb.XGBClassifier or None): XGBoost classifier instance.
        checkpoint_path (Path): Checkpoint file path.
        xgb_model_path (Path): XGBoost model path.
        start_epoch (int): Starting epoch (if resuming).
        config (DotDict): Configuration dictionary.

    Returns:
        tuple: (model, optimizer, xgb_classifier, start_epoch)
    """
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Move optimizer state to the correct device
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

    return model, optimizer, xgb_classifier, start_epoch


def main():
    args = parse_arguments()
    config = load_config_and_setup_device()

    # Paths for saving models and checkpoints
    model_path = Path(config.model_path)
    xgb_model_path = model_path.with_suffix('.xgb')
    checkpoint_path = model_path.with_suffix('.ckpt')
    model_path.parent.mkdir(exist_ok=True)

    # Setup wandb and possibly resume
    wandb_run_id, start_epoch = setup_wandb(config, checkpoint_path)

    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])

    # Load training data
    train_dataloader = load_image_data(
        args.image_directory,
        config.image_size,
        config.batch_size,
        use_sampler=True,
        transform=train_transform
    )

    # Initialize model and optimizer
    model, optimizer, xgb_classifier = initialize_model_and_optimizer(config, model_path, xgb_model_path)

    # If checkpoint exists, load states to resume training
    model, optimizer, xgb_classifier, start_epoch = resume_training_if_checkpoint_exists(
        model, optimizer, xgb_classifier, checkpoint_path, xgb_model_path, start_epoch, config
    )

    # Display model parameter count
    print("Number of parameters: ", count_parameters(model))
    loss_function = nn.BCELoss()

    # Watch the model with wandb
    wandb.watch(model, log='all')

    # Train Vision Transformer
    print("Training vision transformer...")
    train_vision_transformer(
        model,
        train_dataloader,
        optimizer,
        loss_function,
        total_epochs=config.num_epochs,
        device=config.device,
        model_path=model_path,
        xgb_model_path=xgb_model_path,
        xgb_classifier=xgb_classifier,
        checkpoint_path=checkpoint_path,
        start_epoch=start_epoch
    )

    # Extract features from the trained ViT model
    print("Extracting features from vision transformer...")
    features, labels = extract_features(model, train_dataloader, device=config.device)

    # Train and save XGBoost classifier
    print("Training XGBoost classifier...")
    xgb_classifier = train_xgboost(features, labels)
    xgb_classifier.save_model(str(xgb_model_path))
    print(f"XGBoost classifier saved at {xgb_model_path}")

    # If testing is requested, evaluate on test dataset
    if args.test:
        logger.info(f'Testing data: {args.test_directory}')
        test_dataloader = load_image_data(args.test_directory, config.image_size, config.batch_size)

        # Extract test features
        print("Extracting test features...")
        test_features, test_labels = extract_features(model, test_dataloader, device=config.device)

        # Predict using XGBoost classifier
        y_pred = xgb_classifier.predict(test_features)

        # Compute metrics
        accuracy = accuracy_score(test_labels, y_pred)
        precision = precision_score(test_labels, y_pred)
        recall = recall_score(test_labels, y_pred)
        f1 = f1_score(test_labels, y_pred)

        # Log metrics
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
        wandb.log({
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1
        })

    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
