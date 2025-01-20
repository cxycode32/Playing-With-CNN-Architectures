import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from utils import DEVICE, get_lenet_transform, plot_metrics


# Here are hyperparameters
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 64
OUTPUT_CLASSES = 1000
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCH_NUM = 10
AUX_LOGITS = True


class InceptionNet(nn.Module):
    def __init__(self, input_channels, output_channels, output_classes, aux_logits):
        """Initialize the InceptionNet architecture."""
        super(InceptionNet, self).__init__()
        self.aux_logits = aux_logits
        self.device = DEVICE
        
        # Initial convolutional and pooling layers
        self.conv1 = conv_block(input_channels, output_channels, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Inception blocks (series of convolutional layers)
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.aux1 = AuxiliaryClassifier(512, output_classes) if aux_logits else None
        
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.aux2 = AuxiliaryClassifier(528, output_classes) if aux_logits else None
        
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        
        # Final classification layers
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, output_classes)
        
        self.to(self.device)
    
    def forward(self, x):
        x = x.to(self.device)

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        aux1 = self.aux1(x) if self.aux_logits and self.training else None
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x) if self.aux_logits and self.training else None
        
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return (aux1, aux2, x) if self.aux_logits and self.training else x


def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """Creates a convolutional block with BatchNorm and ReLU activation."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class InceptionBlock(nn.Module):
    """
    Inception module with multiple filter sizes.
    
    Each Inception block contains multiple parallel convolutional operations of different filter sizes,
    which allows the model to capture complex features.
    """
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(InceptionBlock, self).__init__()
        
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)
        
        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1pool, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


class AuxiliaryClassifier(nn.Module):
    """
    Auxiliary classifier for additional gradient signal.

    They make predictions during training and help improve the flow of gradients back to earlier layers.
    This helps to avoid vanishing gradients.
    """
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryClassifier, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.7)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv = conv_block(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_inception():
    """
    Train the InceptionNet model on the CIFAR10 dataset and visualize predictions.
    """

    transform = get_lenet_transform()
    
    # Initialize the model, loss function, and optimizer
    model = InceptionNet(input_channels=INPUT_CHANNELS, output_channels=OUTPUT_CHANNELS, output_classes=OUTPUT_CLASSES, aux_logits=AUX_LOGITS).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Download and load the CIFAR10 dataset
    train_dataset = datasets.CIFAR10(root="./cifar10_datasets", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./cifar10_datasets", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

            if isinstance(outputs, tuple):  # Handle auxiliary output case
                aux1_out, aux2_out, main_out = outputs
                loss_1 = criterion(aux1_out, labels)
                loss_2 = criterion(aux2_out, labels)
                loss_3 = criterion(main_out, labels)
                loss_batch = loss_1 * 0.3 + loss_2 * 0.3 + loss_3 * 0.4
                final_output = main_out  # Use main output for accuracy
            else:
                loss_batch = criterion(outputs, labels)
                final_output = outputs

            loss_epoch.append(loss_batch)

            # Backward pass
            optimizer.zero_grad() # Clear previous gradients
            loss_batch.backward() # Compute gradients
            optimizer.step() # Update weights

            # Calculate accuracy
            _, preds = final_output.max(1)
            correct_predictions = (preds == labels).sum()
            total_samples += labels.size(0)
            acc_batch = float(correct_predictions) / float(images.shape[0])
            acc_epoch.append(acc_batch)

        loss_batch = sum(loss_epoch) / len(loss_epoch)
        train_accuracy = sum(acc_epoch) / len(acc_epoch) * 100

        train_losses.append(loss_batch)
        train_accuracies.append(train_accuracy)

        scheduler.step()

        # Print metrics
        print(f"Epoch [{epoch+1}/{EPOCH_NUM}], "
            f"Loss: {loss_batch:.4f}, "
            f"Accuracy: {train_accuracy:.2f}%, "
        )

    # Before passing train_losses and train_accuracies to plot_metrics, move them to CPU
    train_losses = [loss.detach().cpu().numpy() for loss in train_losses]
    train_accuracies = [acc if isinstance(acc, float) else acc.detach().cpu().numpy() for acc in train_accuracies]

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


if __name__ == "__main__":
    train_inception()