# Playing with CNN Architectures

This repository is a dedicated to experimenting with and implementing different Convolutional Neural Network (CNN) architectures. I created this repo as part of my learning process. If you're still learning like me, hope this repo helps~

The repository will include the following CNN architectures:

- **LeNet** ✅ (Completed)
- **VGG (VGG11, VGG13, VGG16, VGG19)** ✅ (Completed)
- **GoogLeNet/InceptionNet** 🚧 (In Progress)
- **ResNet** 🚧 (In Progress)
- **EfficientNet** 🚧 (In Progress)


## Installation

Clone this repository to your local machine:
```bash
git clone https://github.com/cxycode32/Playing-With-CNN-Architectures.git
cd Playing-With-CNN-Architectures
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### File Structure
```
├── lenet.py               # LeNet implementation and training script
├── vgg.py                 # VGG implementation and training script
├── utils.py               # Utility functions
└── requirements.txt       # Project dependencies
```


## How to Run

To run the LeNet implementation and training script:
```bash
python lenet.py
```

To run the VGG implementation and training script:
```bash
python vgg.py
```


## LeNet Implementation

LeNet is one of the earliest CNNs, designed for handwritten digit recognition on the MNIST dataset. It consists of 3 convolutional layers, 2 fully connected layers, and uses ReLU activation and max-pooling.

### Features:

- Beginner-friendly code with detailed explanations.
- Training pipeline for MNIST digit classification.
- Utilities for visualizing data (images and labels) and predictions.

### Visualization:

#### Data (Images and Labels) Visualization
Helper function to visualize a batch of images and labels from the DataLoader.
```
# utils.py

def visualize_data(loader):
```
![Data Visualization](./assets/data_visualization.png)

#### Training Loss and Accuracy
Helper function to visualize training and validation loss and accuracy.
```
# utils.py

def plot_metrics(train_loss, train_acc, val_loss=None, val_acc=None):
```
![Training Loss And Accuracy](./assets/training_loss_acc_lenet.png)

#### Predictions Visualization
Helper function to visualize predictions on test data.
```
# utils.py

def visualize_pred(model, loader):
```
![Predictions Visualization](./assets/pred_visualization.png)


## VGG Implementation

VGG networks are deep CNN architectures that significantly improved performance in large-scale image recognition tasks. The key idea behind VGG is the use of small (3x3) convolutional kernels stacked in depth to learn hierarchical features.

### Features:

- Beginner-friendly code with detailed explanations.
- Multiple deep architectures (VGG11, VGG13, VGG16, VGG19) with increasing layers.
- Utilities for visualizing feature maps and embeddings.

### Training VGG11/VGG13/VGG16/VGG19

When training the model, you may realize some take longer to train, this is caused by:
- **Number of Parameters:** More parameters mean more computations for both forward and backward passes.
- **Network Depth:** Deeper networks equal more layers to propagate the data through and more gradients to compute.
- **Size of Input Data:** Larger inputs require more processing time.
- **Batch Size:** Larger batch sizes can lead to faster training per epoch but also require more memory, which may slow down the training process if the hardware is not sufficient.
- **Optimization Algorithm:** Your hyperparameters can impact training speed.
Of course there are many other factors that affect the model training time, you can do some research on your own if you're interested.

#### Sample Metrics Training VGG11
```
Epoch [1/10], Loss: 2.2400, Accuracy: 21.09%, 
Epoch [2/10], Loss: 1.7921, Accuracy: 30.28%, 
Epoch [3/10], Loss: 1.4089, Accuracy: 48.53%, 
Epoch [4/10], Loss: 1.1012, Accuracy: 61.25%, 
Epoch [5/10], Loss: 0.9006, Accuracy: 68.64%, 
Epoch [6/10], Loss: 0.7754, Accuracy: 73.21%, 
Epoch [7/10], Loss: 0.7213, Accuracy: 75.50%, 
Epoch [8/10], Loss: 0.5226, Accuracy: 82.16%, 
Epoch [9/10], Loss: 0.4630, Accuracy: 84.17%, 
Epoch [10/10], Loss: 0.4259, Accuracy: 85.39%, 
Test Accuracy: 81.10%
```

#### Sample Metrics Training VGG13
```
Epoch [1/10], Loss: 2.3759, Accuracy: 19.08%, 
Epoch [2/10], Loss: 1.8181, Accuracy: 30.06%, 
Epoch [3/10], Loss: 1.4391, Accuracy: 47.37%, 
Epoch [4/10], Loss: 1.1278, Accuracy: 60.62%, 
Epoch [5/10], Loss: 0.9280, Accuracy: 67.74%, 
Epoch [6/10], Loss: 0.8890, Accuracy: 70.18%, 
Epoch [7/10], Loss: 0.7331, Accuracy: 75.11%, 
Epoch [8/10], Loss: 0.5394, Accuracy: 81.59%, 
Epoch [9/10], Loss: 0.4844, Accuracy: 83.35%, 
Epoch [10/10], Loss: 0.4609, Accuracy: 84.10%, 
Test Accuracy: 79.57%
```

#### Sample Metrics Training VGG16
```
Epoch [1/10], Loss: 2.3816, Accuracy: 15.63%, 
Epoch [2/10], Loss: 1.9589, Accuracy: 23.49%, 
Epoch [3/10], Loss: 1.6009, Accuracy: 40.05%, 
Epoch [4/10], Loss: 1.3054, Accuracy: 52.90%, 
Epoch [5/10], Loss: 1.0569, Accuracy: 62.87%, 
Epoch [6/10], Loss: 0.9254, Accuracy: 67.97%, 
Epoch [7/10], Loss: 0.8348, Accuracy: 71.34%, 
Epoch [8/10], Loss: 0.6412, Accuracy: 77.79%, 
Epoch [9/10], Loss: 0.5925, Accuracy: 79.63%, 
Epoch [10/10], Loss: 0.5684, Accuracy: 80.55%, 
Test Accuracy: 78.64%
```

#### Sample Metrics Training VGG19
```
Epoch [1/10], Loss: 2.3364, Accuracy: 17.01%, 
Epoch [2/10], Loss: 1.8138, Accuracy: 31.05%, 
Epoch [3/10], Loss: 1.4842, Accuracy: 45.14%, 
Epoch [4/10], Loss: 1.2772, Accuracy: 54.00%, 
Epoch [5/10], Loss: 1.1679, Accuracy: 59.20%, 
Epoch [6/10], Loss: 0.9875, Accuracy: 65.43%, 
Epoch [7/10], Loss: 0.8697, Accuracy: 69.84%, 
Epoch [8/10], Loss: 0.6843, Accuracy: 76.10%, 
Epoch [9/10], Loss: 0.6289, Accuracy: 78.02%, 
Epoch [10/10], Loss: 0.6040, Accuracy: 78.93%, 
Test Accuracy: 77.77%
```

### Visualization:

#### Feature Maps Visualization
Helper function to visualize the feature maps of selected CNN layer.
```
# utils.py

def visualize_feature_maps(model, image, target_layers):
```

#### Embeddings Visualization
Helper function to visualize high-dimensional feature space.
```
# utils.py

def visualize_embeddings(model, dataloader, num_samples=1000, method='tsne')
```


## GoogLeNet / InceptionNet Implementation

GoogLeNet/InceptionNet is a deep CNN architecture that introduced the Inception module, allowing efficient multi-scale feature extraction while keeping the computational cost reasonable. The model is designed for image classification tasks and includes auxiliary classifiers to improve gradient flow and help with training stability.

### Features:

- **Inception Modules:** The network leverages multiple filter sizes in parallel to capture diverse spatial features.
- **Auxiliary Classifiers:** Helps with gradient propagation during training and regularization.
- **Adaptive Average Pooling:** Reduces feature maps to a single value per channel before classification.
- **Batch Normalization & ReLU Activations:** Used for stable training and improved convergence.
- **Dropout (0.4 probability):** Prevents overfitting in the final fully connected layer.

### Visualization:

#### Training Loss and Accuracy
Helper function to visualize training and validation loss and accuracy.
```
# utils.py

def plot_metrics(train_loss, train_acc, val_loss=None, val_acc=None):
```
![Training Loss And Accuracy](./assets/training_loss_acc_gnet.png)


## Future Plans

The repository will be expanded to include more CNN architectures such as:

- **GoogLeNet/InceptionNet:** Focuses on computational efficiency.
- **ResNet:** Introduces residual connections for training deeper networks.
- **EfficientNet:** Optimizes both accuracy and computational cost.


## Acknowledgments

This implementation is inspired by the ![Machine-Learning-Collection](https://github.com/aladdinpersson/Machine-Learning-Collection) repository by ![aladdinpersson](https://github.com/aladdinpersson). The code structure and visualization techniques are adapted and extended from this collection.