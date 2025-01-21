import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from utils import DEVICE, get_vgg_transform, plot_training_metrics, visualize_feature_maps, visualize_embeddings


# Here are hyperparameters
INPUT_CHANNELS = 3
OUTPUT_CLASSES = 1000
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCH_NUM = 10

# Dictionary defining various VGG architectures
# Each architecture specifies the number of filters and positions of max pooling layers ("M").
VGG_TYPES = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, architecture, input_channels, output_classes):
        """Initialize the VGG architecture.

        Args:
            architecture (str): The type of VGG architecture (e.g., VGG16, VGG19).
            in_channels (int): Number of input channels (e.g., 3 for RGB images).
            num_classes (int): Number of output classes for classification.
        """
        super(VGG, self).__init__()
        self.input_channels = input_channels

        # Create the convolutional layers based on the architecture
        self.conv_layers = self._create_conv_layers(VGG_TYPES[architecture])

        # Fully connected layers for classification
        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # Fully connected layer with 4096 neurons
            nn.ReLU(),  # Activation function
            nn.Dropout(p=0.5),  # Dropout for regularization
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, output_classes),  # Final layer for class scores
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass defining how input flows through the layers.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of class scores.
        """
        x = self.conv_layers(x)  # Pass through convolutional layers
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers
        x = self.fcs(x)  # Pass through fully connected layers
        return x

    def _create_conv_layers(self, architecture: list) -> nn.Sequential:
        """Creates the convolutional layers based on the specified architecture.

        Args:
            architecture (list): List defining the architecture (e.g., [64, 'M', 128, ...]).

        Returns:
            nn.Sequential: A sequential container of convolutional layers.
        """
        layers = []
        input_channels = self.input_channels

        for x in architecture:
            if isinstance(x, int):  # Convolutional layer
                output_channels = x
                layers += [
                    nn.Conv2d(
                        in_channels=input_channels,  # Input channels from the previous layer
                        out_channels=output_channels,  # Number of output filters
                        kernel_size=3,  # Kernel size for feature extraction
                        stride=1,  # Stride for sliding the kernel
                        padding=1,  # Padding to preserve spatial dimensions
                    ),
                    nn.BatchNorm2d(x),  # Batch normalization for faster convergence
                    nn.ReLU(),  # Activation function
                ]
                input_channels = x
            elif x == "M": # Max Pooling layer
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # Reduces spatial dimensions by half

        return nn.Sequential(*layers)  # Combine all layers into a sequential container


def select_architecture():
    """
    Display available VGG architectures then prompt user to select an architecture.
    """
    print("Available VGG Architectures:")
    for idx, architecture in enumerate(VGG_TYPES.keys()):
        print(f"[{idx}] {architecture}")

    # Prompt user to select an architecture
    selected_architecture = None
    while selected_architecture is None:
        try:
            user_input = input("Select the architecture by index (e.g. 0 for VGG11): ").strip()
            
            # When the input is empty
            if not user_input:
                print("Input cannot be empty. Please enter a valid index.")
                continue

            selected_index = int(user_input)

            # When the input is invalid
            if selected_index < 0 or selected_index >= len(VGG_TYPES):
                print("Invalid input. Please enter a valid index.")
                continue

            selected_architecture = list(VGG_TYPES.keys())[selected_index]
            
            return selected_architecture
        
        except ValueError:
            print("Invalid input. Please enter a valid index.")


def train_vgg():
    """
    Train the VGG model on the CIFAR10 dataset and visualize predictions.
    """
    
    transform = get_vgg_transform()

    architecture = select_architecture()

    # Initialize the model, loss function, and optimizer
    model = VGG(architecture=architecture, input_channels=INPUT_CHANNELS, output_classes=OUTPUT_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Download and load the CIFAR10 dataset
    train_dataset = datasets.CIFAR10(root="./cifar10_datasets", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./cifar10_datasets", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()

    # Visualize feature maps
    image, _ = next(iter(train_loader))  # Get a sample image
    visualize_feature_maps(model, image[0], target_layers=['conv_layers.0'])

    # Visualize embeddings
    visualize_embeddings(model, test_loader, method='tsne')

    train_losses = []
    train_accuracies = []

    # Training loop
    for epoch in range(EPOCH_NUM):
        model.train()
        loss_epoch = []
        acc_epoch = []
        correct_predictions = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            train_loss = criterion(outputs, labels) # Compute loss
            loss_batch = train_loss.item()
            loss_epoch.append(loss_batch)

            # Backward pass
            optimizer.zero_grad() # Clear previous gradients
            train_loss.backward() # Compute gradients
            optimizer.step() # Update weights

            # Calculate accuracy
            _, preds = outputs.max(1)
            correct_predictions = (preds == labels).sum()
            total_samples += labels.size(0)
            acc_batch = float(correct_predictions) / float(images.shape[0])
            acc_epoch.append(acc_batch)

        train_loss = sum(loss_epoch) / len(loss_epoch)
        train_accuracy = sum(acc_epoch) / len(acc_epoch) * 100
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        scheduler.step()

        # Print metrics
        print(f"Epoch [{epoch+1}/{EPOCH_NUM}], "
            f"Loss: {train_loss:.4f}, "
            f"Accuracy: {train_accuracy:.2f}%"
        )

    # Plot metrics after training
    plot_training_metrics(train_losses, train_accuracies)

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_accuracy = correct / total * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    train_vgg()
