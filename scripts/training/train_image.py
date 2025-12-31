"""
Training script for image authenticity module (ResNet-18).

Usage:
    python scripts/training/train_image.py \
        --data_dir data/processed/images \
        --output_dir models/image \
        --epochs 25 \
        --batch_size 32 \
        --lr 1e-4
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
import json
from tqdm import tqdm
from PIL import Image

from src.modules.image.classifier import ResNet18FakeDetector
from src.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


class FaceDataset(Dataset):
    """Dataset for real/fake face images"""

    def __init__(self, data_dir: Path, split: str = "train", transform=None):
        self.data_dir = data_dir / split
        self.transform = transform

        # Load image paths
        self.real_images = list((self.data_dir / "real").glob("*.jpg"))
        self.fake_images = list((self.data_dir / "fake").glob("*.jpg"))

        self.images = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)

        logger.info(f"loaded_dataset", split=split,
                   real=len(self.real_images),
                   fake=len(self.fake_images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(dataloader), correct / total


def main():
    parser = argparse.ArgumentParser(description="Train image authenticity model")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()

    # Setup
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("training_setup", device=device, epochs=args.epochs)

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = FaceDataset(data_dir, "train", train_transform)
    val_dataset = FaceDataset(data_dir, "val", val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Model
    model = ResNet18FakeDetector(pretrained=True).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        logger.info(f"epoch_start", epoch=epoch+1)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        logger.info(f"epoch_complete",
                   epoch=epoch+1,
                   train_loss=train_loss,
                   train_acc=train_acc,
                   val_loss=val_loss,
                   val_acc=val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }
            torch.save(checkpoint, output_dir / "resnet18_best.pth")
            logger.info("model_saved", val_acc=val_acc)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            logger.info("early_stopping", epoch=epoch+1)
            break

    logger.info("training_complete", best_val_acc=best_val_acc)


if __name__ == "__main__":
    main()
