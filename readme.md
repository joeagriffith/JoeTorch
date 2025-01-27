# JoeTorch

A Python library containing useful utilities for testing new Deep Learning algorithms.

## Installation

Install using pip:
```bash
pip install joetorch
```

## Features

- **Dataset Utilities**
  - MNIST dataset loader with validation split and augmentation options
  - PreloadedDataset class for efficient data handling
  - Support for custom datasets

- **Neural Network Components**
  - MLP (Multi-Layer Perceptron) module
  - Convolutional blocks (Encoder/Decoder)
  - Auto-encoder architectures

- **Training Utilities**
  - Learning rate schedulers (Cosine, Step, Flat)
  - Mixed precision training support
  - TensorBoard logging integration
  - Optimized weight decay handling

- **Loss Functions**
  - MSE reconstruction loss
  - BCE reconstruction loss
  - KL divergence loss
  - Smooth L1 loss
  - Negative cosine similarity

- **Feature Analysis**
  - Feature correlation analysis
  - Feature standard deviation metrics
  - Representation analysis tools

## Example Usage

```python
from joetorch.datasets import MNIST
from joetorch.nn import MNIST_AE
from joetorch.optim import get_optimiser, train

# Load MNIST dataset
train_dataset = MNIST(root='datasets/', split='train', val_ratio=0.1, 
                     augment=True, device='cuda')
val_dataset = MNIST(root='datasets/', split='val', val_ratio=0.1, 
                   device='cuda')

# Create model and optimizer
model = MNIST_AE(out_dim=20, mode='cnn').to('cuda')
optimizer = get_optimiser(model, optim='AdamW')

# Train the model
train(model, train_dataset, val_dataset, optimizer, 
      num_epochs=50, batch_size=256)
```

## Requirements

- Python >= 3.7
- PyTorch >= 1.19.2
- NumPy >= 1.19.2

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Joe Griffith (joeagriffith@gmail.com)
