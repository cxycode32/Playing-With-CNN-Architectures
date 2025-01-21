import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn.functional as F
from math import ceil
from utils import DEVICE, get_efficientnet_transform, plot_training_metrics


# Here are hyperparameters
INPUT_CHANNELS = 3
OUTPUT_CLASSES = 1000
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32
EPOCH_NUM = 10
GROUPS = 1  # '1' for normal conv, '0' for depthwise conv
REDUCTION = 3 # Squeeze and excitation
SURVIVAL_PROB = 0.8 # Stochastic depth

BASE_MODEL = [
    # These params define how the EfficientNet model will be structured across its layer.
    # [expansion_ratio, channels, repeats, stride, kernel size]
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

# Dictionary defining different EfficientNet versions and its scaling factor for the network's depth, width, and input resolution.
PHI_VALUES = {
    # alpha, beta, gamma, depth = alpha ** phi
    # "version": (phi value, resolution, drop rate)
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}


class CNNBlock(nn.Module):
    """
    A standard convolutional layer with batch normalization and the SiLU activation function (aka Swish).

    What it does: To apply convolution, normalize the output, then apply a non-linear activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=GROUPS):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        # Apply Conv -> BatchNorm -> SiLU activation
        return self.silu(self.bn(self.cnn(x)))


class SqueezeExcitation(nn.Module):
    """
    As the name suggests, it performs squeeze and excitation to recalibrate channel-wise feature responses.

    What it does:
    - To globally averages the feature map, then applies two 1x1 convs and a Sigmoid function to generate channel-wise attention.
    - The output is multiplied by the original feature map to recalibrate it.
    """
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Conv2d(in_channels, reduced_dim, 1),  # Reduce channels
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),  # Rebuild original channels
            nn.Sigmoid(),  # Normalize with sigmoid
        )

    def forward(self, x):
        # Recalibrate the input feature map with SE attention
        return x * self.se(x)


class InvertedResidualBlock(nn.Module):
    """
    The core building block, it uses a lightweight depthwise separable conv after an expansion.

    What it does:
    - Expansion: Increases the number of channels to improve representational power.
    - Depthwise Convolutional: Applies a separate filter for each input channel.
    - Squeeze and Excitation: Recalibrates the channel-wise importance.
    - Residual Connection: If the input and output have the same dimensions, it adds a skip connection.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        expand_ratio,
        reduction=REDUCTION,
        survival_prob=SURVIVAL_PROB,
    ):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        # Expansion convolution, if needed
        if self.expand:
            self.expand_conv = CNNBlock(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1,)

        # Depthwise convolution followed by Squeeze and Excitation and pointwise convolution
        self.conv = nn.Sequential(
            CNNBlock(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim,),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def stochastic_depth(self, x):
        """
        Stochastic depth is a technique used to improve training efficiency and prevent overfitting.

        How it does it:
        It randomly skips/drops certain layers during the training phase.
        Meaning, some layers are not executed.
        So, the network is 'thinner'.
        """
        if not self.training:
            return x

        # survival_prob determines the likelihood of a layer being kept during a forward pass
        # binary_tensor generates a random value 0 ~ 1, if binary_tensor > survival prob, drop the layer
        binary_tensor = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob)
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        # Apply expansion conv (if needed)
        x = self.expand_conv(inputs) if self.expand else inputs

        # Apply stochastic depth, then either residual connection or just the output
        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    """EfficientNet architecture that adapts to different model versions."""
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        
        # Calculate scaling factors for depth, width, and dropout rate
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)

        # Initialize layers
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(last_channels, num_classes),)

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        """ Calculate scaling factors based on the model version. """
        phi, res, drop_rate = PHI_VALUES[version]
        depth_factor = alpha**phi
        width_factor = beta**phi
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels):
        """ Build the feature extraction layers (from input to last channels). """
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in BASE_MODEL:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,  # if k=1 use pad=0, k=3 use pad=1, k=5 use pad=2
                    )
                )
                in_channels = out_channels

        features.append(CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0))

        return nn.Sequential(*features)

    def forward(self, x):
        """ Define forward pass for the model """
        x = self.pool(self.features(x))  # Extract features
        return self.classifier(x.view(x.shape[0], -1))  # Flatten and pass to classifier


def select_version():
    """
    Display available EfficientNet versions then prompt user to select a version.
    """
    print("Available EfficientNet versions:")
    for idx, version in enumerate(PHI_VALUES.keys()):
        print(f"[{idx}] {version}")

    # Prompt user to select a version
    selected_version = None
    while selected_version is None:
        try:
            user_input = input("Select the version by index: ").strip()
            
            # When the input is empty
            if not user_input:
                print("Input cannot be empty. Please enter a valid index.")
                continue

            selected_index = int(user_input)

            # When the input is invalid
            if selected_index < 0 or selected_index >= len(PHI_VALUES):
                print("Invalid input. Please enter a valid index.")
                continue

            selected_version = list(PHI_VALUES.keys())[selected_index]

            return selected_version
        
        except ValueError:
            print("Invalid input. Please enter a valid index.")


def train_efficientnet():
    """
    Train the EfficientNet on the CIFAR100 dataset and visualize predictions.
    """

    version = select_version()
    phi_value, resolution, drop_rate = PHI_VALUES[version]
    transform = get_efficientnet_transform(resolution)
    print(f"Using EfficientNet-{version} with input resolution {resolution}x{resolution}")

    # Initialize the model, loss function, and optimizer
    model = EfficientNet(version=version, num_classes=OUTPUT_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Download and load the CIFAR100 dataset
    train_dataset = datasets.CIFAR100(root="./cifar100_datasets", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR100(root="./cifar100_datasets", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_losses, test_loss = [], []
    train_accuracies, test_accuracies = [], []

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
    test_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = correct / total * 100
    print(f"Test Loss: {test_loss:.4f}, "
        f"Test Accuracy: {test_accuracy:.2f}%"
    )


if __name__ == "__main__":
    train_efficientnet()
