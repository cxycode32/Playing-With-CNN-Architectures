# Playing with CNN Architectures

This repository is a dedicated to experimenting with and implementing different Convolutional Neural Network (CNN) architectures. I created this repo as part of my learning process. If you're still learning like me, hope this repo helps~

The repository will include the following CNN architectures:

- LeNet (Completed)
- VGG (Work in Progress)
- GoogLeNet/InceptionNet (Planned)
- ResNet (Planned)
- EfficientNet (Planned)

Currently, the implementation of LeNet is complete, along with a training pipeline and visualization utilities.


## LeNet Implementation

LeNet is one of the earliest convolutional neural networks, designed primarily for handwritten digit recognition on the MNIST dataset. It consists of three convolutional layers, two fully connected layers, and uses ReLU activation and max-pooling.

### Features:

- Beginner-friendly code with detailed explanations.
- Flexible architecture with adjustable output classes.
- Training pipeline for MNIST digit classification.
- Utilities for visualizing data and predictions.


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
├── lenet.py               # Implementation of LeNet with training script
├── utils.py               # Utility functions
├── requirements.txt       # Project dependencies
└── .gitignore             # Ignored files for Git
```


## How to Run

Run the LeNet training script:
```bash
python lenet.py
```


## Future Plans

The repository will be expanded to include more advanced architectures such as:

- **VGG:** Known for its depth and simplicity.
- **GoogLeNet/InceptionNet:** Focuses on computational efficiency.
- **ResNet:** Introduces residual connections for training deeper networks.
- **EfficientNet:** Optimizes both accuracy and computational cost.

Stay tuned for updates!


## Contribution

Feel free to fork this repository and submit pull requests to improve the project or add new features.


## License

This project is licensed under the MIT License.


## Acknowledgments

The MNIST dataset, provided by Yann LeCun et al., is used for training and evaluation.



Happy coding and learning!