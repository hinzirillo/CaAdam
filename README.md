# CaAdam: Connection Aware Adam Optimizer

# CaAdam: Connection-Aware Adam Optimizer

[![PyPI version](https://img.shields.io/pypi/v/caadam.svg)](https://pypi.org/project/caadam/)

CaAdam is a novel optimization approach that enhances the Adam optimizer by incorporating neural network architectural awareness through connection-based scaling strategies. It consistently achieves faster convergence and better minima compared to standard Adam across various architectures and tasks.
Paper is available [here](https://arxiv.org/abs/2410.24216).

## Key Features

- Drop-in replacement for Adam optimizer in Keras
- Three scaling strategies for different architectural needs:
  - Additive MinMaxMedian scaling
  - Multiplicative MinMaxMedian scaling
  - Depth-based scaling
- Compatible with standard Keras model training workflows
- Improved convergence rates and final model performance
- Write in pure keras3 and compatible with TensorFlow, JAX, and PyTorch

## Installation

```bash
pip install caadam
```

## Quick Start

```python
from caadam import CaAdam, MultiplicativeMinMaxMedianConnectionScaling

# Initialize scaling strategy
strategy = MultiplicativeMinMaxMedianConnectionScaling()

# Create optimizer
optimizer = CaAdam(
    learning_rate=0.001,
    scaling_strategy=strategy
)

# Use with Keras model
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

## Available Scaling Strategies

### 1. Multiplicative MinMaxMedian Scaling
```python
from caadam import MultiplicativeMinMaxMedianConnectionScaling

strategy = MultiplicativeMinMaxMedianConnectionScaling()
optimizer = CaAdam(learning_rate=0.001, scaling_strategy=strategy)
```
Best for general use cases and simpler architectures. Consistently shows strong performance across different tasks.

### 2. Additive MinMaxMedian Scaling
```python
from caadam import AdditiveMinMaxMedianConnectionScaling

strategy = AdditiveMinMaxMedianConnectionScaling()
optimizer = CaAdam(learning_rate=0.001, scaling_strategy=strategy)
```
Suitable for networks with moderate depth and regular connectivity patterns.

### 3. Depth-based Scaling
```python
from caadam import DepthConnectionScaling

strategy = DepthConnectionScaling()
optimizer = CaAdam(learning_rate=0.001, scaling_strategy=strategy)
```
Particularly effective for deeper architectures and complex networks.

## Complete Example

```python
import keras
from caadam import CaAdam, MultiplicativeMinMaxMedianConnectionScaling

# Prepare data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Initialize CaAdam
strategy = MultiplicativeMinMaxMedianConnectionScaling()
optimizer = CaAdam(learning_rate=0.001, scaling_strategy=strategy)

# Compile and train
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
```

## Performance Benefits

Based on extensive empirical evaluation, CaAdam shows significant improvements over standard Adam:

- Classification tasks: Up to 5.97% accuracy improvement on CIFAR-100
- Regression tasks: Up to 2.87% RMSE reduction
- Training time: Up to 30% reduction in some architectures
- Convergence: Typically requires fewer epochs to reach optimal performance

## Citation

If you use CaAdam in your research, please cite:

```
@article{genet2024caadam,
  title={CaAdam: Improving Adam optimizer using connection aware methods},
  author={Genet, Remi and Inzirillo, Hugo},
  journal={arXiv preprint arXiv:2410.24216},
  year={2024}
}
```

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
