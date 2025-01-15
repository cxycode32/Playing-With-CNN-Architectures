import torch
from torchvision import transforms
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transform():
    """Return the data augmentation pipeline."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def visualize_data(loader):
    """
    Helper function to visualize a batch of images and labels from the DataLoader.

    Args:
        loader (DataLoader): DataLoader to sample images and labels from.
    """
    images, labels = next(iter(loader))
    fig, axes = plt.subplots(1, 6, figsize=(15, 5))
    for i in range(6):
        axes[i].imshow(images[i].squeeze(), cmap="gray")
        axes[i].set_title(f"Label: {labels[i].item()}")
        axes[i].axis("off")
    plt.show()


def visualize_pred(model, loader):
    """
    Helper function to visualize predictions on test data.

    Args:
        loader (DataLoader): DataLoader to sample images and labels from.
    """
    images, labels = next(iter(loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)
    print("Visualizing predictions on test data...")

    # Move images and labels to CPU for visualization
    images, labels, predictions = images.cpu(), labels.cpu(), predictions.cpu()

    fig, axes = plt.subplots(1, 6, figsize=(15, 5))
    for i in range(6):
        axes[i].imshow(images[i].squeeze(), cmap="gray")
        axes[i].set_title(f"Pred: {predictions[i].item()} | True: {labels[i].item()}")
        axes[i].axis("off")
    plt.show()
