import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from utils import DEVICE, get_lenet_transform, plot_metrics, visualize_data, visualize_pred, visualize_feature_maps


# Here are hyperparameters
OUTPUT_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCH_NUM = 10


class LeNet(nn.Module):
    def __init__(self, output_classes):
        """
        Initialize the LeNet architecture with flexibility for custom number of output classes.
        
        Args:
            output_classes (int): Number of output classes for classification.
        """
        super(LeNet, self).__init__()

        # Activation function for non-linearity (helps the model learn complex features)
        self.relu = nn.ReLU()

        # Pooling layer to downsample feature maps and reduce computational cost
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1st convolutional layer: extracts 6 feature maps
        self.conv1 = nn.Conv2d(
            in_channels=1,  # Grayscale image has 1 channel
            out_channels=6,  # Number of filters (feature maps)
            kernel_size=5,  # Size of each filter is 5x5
            stride=1,  # Filters move 1 pixel at a time
            padding=2,  # Padding ensures the output has the same spatial size as input
        )

        # 2nd convolutional layer: extracts 16 feature maps
        self.conv2 = nn.Conv2d(
            in_channels=6,  # Takes 6 input feature maps from conv1
            out_channels=16,  # Outputs 16 feature maps
            kernel_size=5,
            stride=1,
            padding=0,  # No padding here
        )
        
        # 3rd convolutional layer: extracts 120 feature maps and reduces spatial dimentions
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=5,
            stride=1,
            padding=0,
        )

        # Fully connected layers for classification
        # Reduces 120 features to 84
        self.linear1 = nn.Linear(120, 84)

        # Final layer
        # Outputs predictions for each class
        self.linear2 = nn.Linear(84, output_classes)

    def forward(self, x):
        """
        Forward pass defining how input flows through the layers.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes]
        """
        x = self.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        x = self.pool(x)  # Downsample using max pooling
        x = self.relu(self.conv2(x))  # Apply second convolution and ReLU activation
        x = self.pool(x)  # Downsample again
        x = self.relu(self.conv3(x))  # Apply third convolution and ReLU activation

        # Flatten the tensor into a 1D vector for input into fully connected layers
        x = x.view(x.size(0), -1)
        x = self.relu(self.linear1(x))  # Apply first fully connected layer and ReLU activation
        x = self.linear2(x)  # Apply second fully connected layer to produce output logits
        return x


def train_lenet():
    """
    Train the LeNet model on the MNIST dataset and visualize predictions.
    """
    
    transform = get_lenet_transform()

    # Initialize the model, loss function, and optimizer
    model = LeNet(output_classes=OUTPUT_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Download and load the MNIST dataset
    train_dataset = datasets.MNIST(root="./mnist_dataset", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./mnist_dataset", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Visualize images and labels
    visualize_data(train_loader)

    # Visualize feature maps
    image, _ = next(iter(train_loader))  # Get a batch of images
    visualize_feature_maps(model, image[0], target_layers=['conv_layers.0'])  # Adjusted target layer

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

        # Print metrics
        print(f"Epoch [{epoch+1}/{EPOCH_NUM}], "
            f"Loss: {train_loss:.4f}, "
            f"Accuracy: {train_accuracy:.2f}%, "
        )

    # Plot metrics after training
    plot_metrics(train_losses, train_accuracies)

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

    # Visualize predictions
    visualize_pred(model, test_loader)


if __name__ == "__main__":
    train_lenet()