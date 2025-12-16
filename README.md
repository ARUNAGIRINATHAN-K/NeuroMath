<div align="center">

![Neural Networks Banner](neuromath.png)

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

[![GitHub stars](https://img.shields.io/github/stars/ARUNAGIRINATHAN-K/NeuroMath?style=social)](https://github.com/ARUNAGIRINATHAN-K/NeuroMath/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/ARUNAGIRINATHAN-K/NeuroMath?style=social)](https://github.com/ARUNAGIRINATHAN-K/NeuroMath/network/members)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](#-contributing)

---

*Artificial Neural Networks (ANNs) are powerful computational models inspired by the human brain, and their strength lies in the mathematics that governs how they learn and make predictions. At their core, ANNs use linear algebra to represent data as vectors and matrices, and apply weighted transformations to propagate information through layers. Non-linear activation functions introduce complexity, enabling networks to learn relationships beyond simple linear patterns.Learning in ANNs is driven by calculus, specifically gradient-based optimization. Using backpropagation, networks compute partial derivatives of the loss function with respect to each weight, adjusting parameters to minimize error. Probability and statistics further support ANN behavior by defining loss functions, modeling uncertainty, and improving generalization. Overall, the mathematics of ANNs forms the foundation for training stable, accurate, and scalable deep learning models.*

---

## 🛠️ Tech Stack & Tools

| Category | Tool | Badge | Purpose |
|----------|------|-------|---------|
| **Language** | Python 3.9+ | [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org) | Primary programming language |
| **Core Library** | NumPy | [![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) | Numerical computing & matrix operations |
| **Visualization** | Matplotlib | [![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org) | Data visualization & plotting |
| **ML Utilities** | Scikit-Learn | [![Scikit Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org) | Dataset loading & preprocessing |
| **Notebook** | Jupyter | [![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org) | Interactive development environment |
| **Editor** | VS Code | [![VS Code](https://img.shields.io/badge/VS_Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white)](https://code.visualstudio.com) | Code editor & IDE |
| **Version Control** | Git | [![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)](https://git-scm.com) | Source code management |
| **Repository** | GitHub | [![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com) | Code hosting & collaboration |
| **Framework** | TensorFlow | [![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org) | Benchmarking & comparison only |

---

## Project Structure

| # | Project | Folder | Key Concepts |
|---|---------|--------|--------------|
| 01 | **Perceptron Learning Rule** | [01_perceptron_learning](01_nn_from_scratch) | Linear separability, weight updates |
| 02 | **XOR with MLP** | [02_xor_mlp](02_xor_classification) | Non-linearity, backpropagation |
| 03 | **MNIST Digit Recognition** | [03_mnist_digit_recognition](03_mnist_digit_recognition) | Multi-class classification, softmax |
| 04 | **Neural Network Visualizer** | [04_nn_visualizer](04_nn_visualizer) | Training dynamics, weight evolution |
| 05 | **Custom Dataset ANN** | [05_custom_dataset_ann](05_custom_dataset_ann) | Tabular data, label encoding |
| 06 | **Loss Surface Visualization** | [06_loss_landscape](06_loss_landscape) | Loss contours, optimization geometry |
| 07 | **Backpropagation Simulator** | [07_backprop_simulator](07_backprop_simulator) | Chain rule, matrix calculus |
| 08 | **Activation Function Analysis** | [08_activation_function_analysis](08_activation_function_analysis) | ReLU vs. Sigmoid vs. Tanh |
| 09 | **Dropout Regularization** | [09_dropout_regularization](09_dropout_regularization) | Overfitting prevention |
| 10 | **Time Series Forecasting** | [10_time_series_ann](10_time_series_ann) | Sliding window, ANN regression |

---
</div>

## Architecture Overview

```
                           ┌────────────────────────┐
                           │ Mathematics of ANN     │
                           └─────────────┬──────────┘
                                         │
        ┌────────────────────────────────┼─────────────────────────────────┐
        │                                │                                 │
        ▼                                ▼                                 ▼
┌────────────────┐             ┌────────────────────┐             ┌────────────────────┐
│  Core Building │             │ Training &         │             │ Advanced Concepts  │
│  Blocks        │             │ Optimization       │             │ & Experiments      │
└───────┬────────┘             └─────────┬──────────┘             └─────────┬──────────┘
        │                                │                                  │
        ▼                                ▼                                  ▼
┌──────────────────┐          ┌──────────────────────┐          ┌──────────────────────────┐
│ • Perceptron     │          │ • Gradient Descent   │          │ • Backprop Simulator     │
│ • Activation     │          │ • Loss Functions     │          │ • Dropout Regularization │
│   Functions      │          │ • Loss Landscape Viz │          │ • Time Series Forecast   │
└──────────────────┘          └──────────────────────┘          │ • MNIST Recognition      │
                                                                └──────────────────────────┘
```

## Quick Start

### Prerequisites

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![pip](https://img.shields.io/badge/pip-latest-green?logo=pypi&logoColor=white)

**Recommended Knowledge:**
- Basic linear algebra and calculus
- Python programming (NumPy basics)
- Understanding of gradient descent

### Installation

```bash
# Clone the repository
git clone https://github.com/ARUNAGIRINATHAN-K/NeuroMath.git
cd NeuroMath

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

### Dependencies

```txt
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
tensorflow>=2.8.0  # For comparison only
jupyter>=1.0.0
```

<div align="center">
   
---

### Activation Functions

*Activation functions introduce **non-linearity** into neural networks, enabling them to learn complex patterns. Without them, networks would behave like linear regression regardless of depth.*

#### Common Activation Functions

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| **Sigmoid** | $f(x) = \frac{1}{1 + e^{-x}}$ | (0, 1) | Binary classification output |
| **Tanh** | $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | (-1, 1) | Hidden layers (zero-centered) |
| **ReLU** | $f(x) = \max(0, x)$ | [0, ∞) | Hidden layers (most popular) |
| **Softmax** | $P(y=i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$ | (0, 1) | Multi-class classification |

---


[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/arunagirinathan-k)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ARUNAGIRINATHAN-K/NeuroMath)

---


**[⬆ Back to Top](#-neural-networks-from-scratch)**


</div>
