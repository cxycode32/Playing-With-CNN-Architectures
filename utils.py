import torch
from torchvision import transforms
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_lenet_transform():
    """Return the data augmentation pipeline for LeNet."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def get_vgg_transform():
    """Return the data augmentation pipeline for VGG."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_resnet_transform():
    """Return the data augmentation pipeline for ResNet."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_efficientnet_transform(resolution):
    """Return the data augmentation pipeline for EfficientNet."""
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        model (torch.nn.Module): Trained model.
        loader (DataLoader): DataLoader to sample images and labels from.
    """
    images, labels = next(iter(loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)

    # Move images and labels to CPU for visualization
    images, labels, predictions = images.cpu(), labels.cpu(), predictions.cpu()

    fig, axes = plt.subplots(1, 6, figsize=(15, 5))
    for i in range(6):
        axes[i].imshow(images[i].squeeze(), cmap="gray")
        axes[i].set_title(f"Pred: {predictions[i].item()} | True: {labels[i].item()}")
        axes[i].axis("off")
    plt.show()


def visualize_feature_maps(model, image, target_layers):
    """
    Visualizes feature maps of selected CNN layer.
    This helps you understand what features the convolutional layers are learning.
    
    Args:
        model (torch.nn.Module): Pre-trained model.
        image (torch.Tensor): Input image tensor (preprocessed).
        target_layers (list of str): List of layer names to visualize.
    """
    activations = {}
    
    # Hook to capture intermediate activations
    def hook_fn(module, input, output):
        activations[module] = output.detach()
    
    # Register hooks on multiple layers
    hooks = []
    for name, layer in model.named_modules():
        if name in target_layers:
            hooks.append(layer.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        _ = model(image.unsqueeze(0).to(DEVICE))  # Add batch dimension
    
    for hook in hooks:
        hook.remove()
    
    # Convert feature maps to numpy for visualization
    for layer_name, feature_maps in activations.items():
        feature_maps = feature_maps.squeeze(0).cpu().numpy()
        num_maps = min(feature_maps.shape[0], 6)
        
        fig, axes = plt.subplots(1, num_maps, figsize=(15, 5))
        fig.suptitle(f'Feature Maps of Layer: {layer_name}', fontsize=14)
        
        for i in range(num_maps):
            axes[i].imshow(feature_maps[i], cmap='gray')
            axes[i].axis('off')
        plt.show()


def visualize_embeddings(model, dataloader, num_samples=1000, method='tsne'):
    """
    Uses t-SNE or UMAP to visualize high-dimensional feature space.
    This helps you to see how the model clusters different classes.
    
    Args:
        model (torch.nn.Module): Trained CNN model.
        dataloader (torch.utils.data.DataLoader): DataLoader for dataset.
        num_samples (int): Number of samples to visualize.
        method (str): Either 'tsne' or 'umap'.
    """
    model.eval()
    features, labels = [], []
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            if len(features) >= num_samples:
                break
            output = model(images.to(DEVICE))[:, :-1]  # Use penultimate layer
            features.append(output.cpu().numpy())
            labels.append(targets.cpu().numpy())
    
    features = np.vstack(features)
    labels = np.hstack(labels)
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Invalid method. Choose 'tsne' or 'umap'.")
    
    reduced_features = reducer.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='jet', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title(f"{method.upper()} Visualization of Feature Space")
    plt.show()


def plot_training_metrics(train_loss, train_acc, val_loss=None, val_acc=None):
    """
    Helper function to visualize training and validation loss and accuracy.

    Args:
        train_loss: Training loss.
        train_acc: Training accuracy.
        val_loss: Validation loss.
        val_acc: Validation accuracy.
    """
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'r-', label='Training Loss')
    if val_loss:
        plt.plot(epochs, val_loss, 'b-', label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'r-', label='Training Accuracy')
    if val_acc:
        plt.plot(epochs, val_acc, 'b-', label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_gnet_metrics(train_loss, train_acc, val_loss=None, val_acc=None):
    """
    Helper function to visualize training and validation loss and accuracy.
    Like the plot_metrics function but it's for GoogLeNet/InceptionNet.

    Args:
        train_loss: Training loss.
        train_acc: Training accuracy.
        val_loss: Validation loss.
        val_acc: Validation accuracy.
    """
    # Move tensors to CPU and convert them to NumPy arrays if they are on the GPU
    if torch.is_tensor(train_loss):
        train_loss = train_loss.cpu().numpy()
    if torch.is_tensor(train_acc):
        train_acc = train_acc.cpu().numpy()
    if val_loss is not None and torch.is_tensor(val_loss):
        val_loss = val_loss.cpu().numpy()
    if val_acc is not None and torch.is_tensor(val_acc):
        val_acc = val_acc.cpu().numpy()

    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'r-', label='Training Loss')
    if val_loss is not None:
        plt.plot(epochs, val_loss, 'b-', label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'r-', label='Training Accuracy')
    if val_acc is not None:
        plt.plot(epochs, val_acc, 'b-', label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_resnet_metrics(resnet_models, epoch_num, train_accuracies):
    """
    Helper function to visualize training accuracy for all ResNet variants to evaluate their performance.

    Args:
        resnet_models: ResNet model.
        epoch_num: Epoch number.
        train_accuracies: Training accuracy.
    """
    plt.figure(figsize=(10, 5))
    for name in resnet_models.keys():
        plt.plot(range(1, epoch_num + 1), train_accuracies[name], label=f"{name}")
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy (%)")
    plt.title("Training Accuracy Comparison")
    plt.legend()
    plt.show()