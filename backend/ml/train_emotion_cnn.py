
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmotionDataset(Dataset):
    """Custom PyTorch Dataset for emotion data."""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # The data is expected to be in (N, H, W) format. Add a channel dimension.
        image = self.images[idx]
        image = np.expand_dims(image, axis=-1) # Becomes (H, W, C)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

class SimpleCNN(nn.Module):
    """A simple CNN for emotion classification."""
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Assuming input images are 48x48, after 3 max-pooling layers, the size is 6x6
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 6 * 6, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def train(args):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- 1. Load Data ---
    logging.info(f"Loading data from {args.data_path}...")
    try:
        data = np.load(args.data_path)
        # Assuming the npz file has keys 'X' for images and 'y' for labels
        images = data['X']
        labels = data['y']
        logging.info(f"Data loaded successfully. Found {len(images)} samples.")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return

    # --- 2. Create Datasets and DataLoaders ---
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # --- 3. Define Transforms ---
    # Basic transforms for validation
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Transforms with augmentation for training
    if args.augment:
        logging.info("Data augmentation enabled.")
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        logging.info("Data augmentation disabled.")
        train_transform = val_transform

    train_dataset = EmotionDataset(X_train, y_train, transform=train_transform)
    val_dataset = EmotionDataset(X_val, y_val, transform=val_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)

    # --- 4. Initialize Model, Loss, and Optimizer ---
    model = SimpleCNN(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # --- 5. Training Loop ---
    best_val_accuracy = 0.0
    logging.info("Starting training...")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_accuracy = accuracy_score(train_labels, train_preds)
        
        # --- 6. Validation ---
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, val_preds)

        logging.info(
            f"Epoch [{epoch+1}/{args.epochs}], "
            f"Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.4f}"
        )

        # --- 7. Save Best Model ---
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            os.makedirs(args.output_dir, exist_ok=True)
            model_path = os.path.join(args.output_dir, f"emotion_model_v{args.version}.pt")
            torch.save(model.state_dict(), model_path)
            logging.info(f"Model improved. Saved to {model_path}")

    # --- 8. Save Training Metadata ---
    metadata = {
        'version': args.version,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'augmentation': args.augment,
        'final_validation_accuracy': best_val_accuracy
    }
    metadata_path = os.path.join(args.output_dir, f"training_metadata_v{args.version}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Training finished. Best validation accuracy: {best_val_accuracy:.4f}")
    logging.info(f"Training metadata saved to {metadata_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN for emotion classification.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the .npz data file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save trained models.')
    parser.add_argument('--version', type=str, default='0.1', help='Version of the model.')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size.')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of emotion classes.')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation.')
    
    args = parser.parse_args()
    train(args)
