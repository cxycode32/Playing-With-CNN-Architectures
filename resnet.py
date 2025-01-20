import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from utils import DEVICE, get_resnet_transform, plot_resnet_metrics


# Here are hyperparameters
IMAGE_CHANNELS = 3
OUTPUT_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCH_NUM = 10


class block(nn.Module):
    def __init__(self, input_channels, intermediate_channels, identity_downsample=None, stride=1):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(input_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False,)
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, output_classes):
        """Initialize the ResNet architecture."""
        super(ResNet, self).__init__()
        self.input_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, output_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.input_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.input_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.input_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.input_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.input_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet50(image_channels, output_classes):
    return ResNet(block, [3, 4, 6, 3], image_channels, output_classes)


def ResNet101(image_channels, output_classes):
    return ResNet(block, [3, 4, 23, 3], image_channels, output_classes)


def ResNet152(image_channels, output_classes):
    return ResNet(block, [3, 8, 36, 3], image_channels, output_classes)


# Define available ResNet architectures
RESNET_TYPES = {
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152,
}


def select_net():
    """
    Display available ResNet types and prompt user for selection.
    Returns the corresponding ResNet model.
    """
    print("Available ResNet Types:")
    for idx, net_name in enumerate(RESNET_TYPES.keys()):
        print(f"[{idx}] {net_name}")

    while True:
        try:
            user_input = int(input("Select the net type by index: ").strip())
            if user_input < 0 or user_input >= len(RESNET_TYPES):
                print("Invalid input. Try again.")
                continue
            return list(RESNET_TYPES.values())[user_input]
        except ValueError:
            print("Invalid input. Enter a number.")


def train_resnet():
    """
    Train the selected ResNet model.
    """
    transform = get_resnet_transform()

    # Get user-selected ResNet model
    # net = select_net()

    # Download and load the CIFAR10 dataset
    train_dataset = datasets.CIFAR10(root="./cifar10_datasets", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./cifar10_datasets", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Store results
    train_losses, test_losses = {}, {}
    train_accuracies, test_accuracies = {}, {}

    # Train and evaluate each ResNet type
    for name, func in RESNET_TYPES.items():
        print(f"Training {name}......")
    
        model = func(IMAGE_CHANNELS, OUTPUT_CLASSES).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        train_losses[name], train_accuracies[name] = [], []
        test_losses[name], test_accuracies[name] = [], []

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

            train_losses[name].append(train_loss)
            train_accuracies[name].append(train_accuracy)

            scheduler.step()

            # Print metrics
            print(f"Epoch [{epoch+1}/{EPOCH_NUM}], "
                f"Loss: {train_loss:.4f}, "
                f"Accuracy: {train_accuracy:.2f}%, "
            )

        # Evaluate the model
        model.eval()
        correct, total, test_loss = 0, 0, 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        test_accuracy = correct / total * 100

        test_losses[name].append(test_loss / len(test_loader))
        test_accuracies[name].append(test_accuracy)

        print(f"{name} Test Accuracy: {test_accuracy[name]:.2f}%")

    plot_resnet_metrics(RESNET_TYPES, EPOCH_NUM, train_accuracies, test_accuracies)
    

if __name__ == "__main__":
    train_resnet()
