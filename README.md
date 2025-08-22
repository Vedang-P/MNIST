# MNIST Digit Classification with Deep Learning

A comprehensive deep learning project that compares three different neural network architectures for classifying handwritten digits from the MNIST dataset.

## 📋 Project Overview

This project implements and evaluates three distinct deep learning models for the classic MNIST handwritten digit classification task:

1. **Fully Connected Neural Network (FCN)**
2. **Convolutional Neural Network (CNN)**
3. **EfficientNetB3 (Transfer Learning)**

The goal is to demonstrate how different architectural choices impact performance on image classification tasks and provide insights into the effectiveness of various deep learning approaches.

## 🎯 Dataset

- **MNIST Dataset**: 70,000 handwritten digit images (0-9)
- **Training set**: 60,000 images (28×28 pixels, grayscale)
- **Test set**: 10,000 images (28×28 pixels, grayscale)
- **Classes**: 10 digits (0-9)
- **Image format**: Grayscale, 28×28 pixels, pixel values 0-255

## 🏗️ Model Architectures

### 1. Fully Connected Neural Network (FCN)

**Architecture:**
```
Input Layer: 784 neurons (28×28 flattened)
Hidden Layer 1: 256 neurons (ReLU activation)
Hidden Layer 2: 256 neurons (ReLU activation)
Hidden Layer 3: 32 neurons (ReLU activation)
Output Layer: 10 neurons (Softmax activation)
```

**Key Features:**
- Total parameters: 275,306
- Simple feedforward architecture
- Good baseline for comparison

### 2. Convolutional Neural Network (CNN)

**Architecture:**
```
Conv2D: 32 filters (3×3), ReLU activation
MaxPooling2D: 2×2 pooling
BatchNormalization
Conv2D: 64 filters (3×3), ReLU activation
MaxPooling2D: 2×2 pooling
BatchNormalization
Flatten
Dense: 128 neurons (ReLU activation)
Dropout: 0.4
Output: 10 neurons (Softmax activation)
```

**Key Features:**
- Total parameters: 225,418
- Leverages spatial relationships in images
- Includes regularization techniques (BatchNorm, Dropout)

### 3. EfficientNetB3 (Transfer Learning)

**Architecture:**
```
Pre-trained EfficientNetB3 (ImageNet weights)
Frozen EfficientNet layers
Flatten
Dense: 512 neurons (ReLU activation)
Dropout: 0.25
Output: 10 neurons (Softmax activation)
```

**Key Features:**
- Total parameters: 11,575,609 (only 792,074 trainable)
- Pre-trained on ImageNet dataset
- Transfer learning approach

## 🔧 Data Preprocessing

### Standard Preprocessing Steps:
1. **Reshaping**: Converting 28×28 images to appropriate input formats
2. **Normalization**: Scaling pixel values from 0-255 to 0-1 range
3. **One-hot encoding**: Converting labels to categorical format
4. **Channel expansion**: For EfficientNet (converting grayscale to RGB)

### Model-Specific Preprocessing:
- **FCN**: Flatten to 784-dimensional vectors
- **CNN**: Reshape to (28, 28, 1) with single channel
- **EfficientNet**: Resize to (32, 32, 3) and expand to RGB

## 🚀 Training Strategy

### Common Training Parameters:
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Early Stopping**: Monitor validation loss with patience=20
- **Model Checkpointing**: Save best model weights

### Model-Specific Training:
- **FCN**: 30% validation split, batch size 32
- **CNN**: Full test set validation, batch size 32
- **EfficientNet**: Full test set validation, batch size 64

## 📊 Results

| Model | Training Accuracy | Test Accuracy | Parameters | Performance |
|-------|------------------|---------------|------------|-------------|
| FCN | 99.29% | 97.95% | 275K | Good |
| CNN | 99.91% | 99.25% | 225K | **Best** |
| EfficientNetB3 | 28.52% | 29.57% | 11.6M | Poor |

## 📈 Performance Analysis

### Confusion Matrix Results:
- **FCN**: Overall accuracy 97.95% with good precision/recall across all digits
- **CNN**: Overall accuracy 99.25% with excellent precision/recall (mostly >99%)
- **EfficientNet**: Poor performance due to domain mismatch

### Key Performance Insights:
1. **CNN outperforms FCN**: Better test accuracy with fewer parameters
2. **Transfer learning challenge**: EfficientNet struggles with domain adaptation
3. **Overfitting prevention**: Early stopping and regularization techniques effective
4. **Data efficiency**: CNN leverages spatial relationships effectively

## 🔍 Key Insights

### 1. Architectural Impact
- **CNN superiority**: Convolutional layers capture spatial patterns better than fully connected layers
- **Parameter efficiency**: CNN achieves better performance with fewer parameters
- **Feature learning**: CNN learns hierarchical features automatically

### 2. Transfer Learning Challenges
- **Domain mismatch**: EfficientNet pre-trained on natural images struggles with simple handwritten digits
- **Over-parameterization**: Too many parameters for simple MNIST task
- **Feature complexity**: Pre-trained features may be too sophisticated for simple digit recognition

### 3. Regularization Effectiveness
- **Batch Normalization**: Helps with training stability and convergence
- **Dropout**: Prevents overfitting effectively
- **Early Stopping**: Prevents overfitting by monitoring validation loss

### 4. Data Preprocessing Importance
- **Normalization**: Critical for stable training
- **Proper reshaping**: Essential for different model architectures
- **One-hot encoding**: Necessary for multi-class classification

## 🛠️ Technical Implementation

### Dependencies:
```python
tensorflow>=2.0
keras
numpy
matplotlib
scikit-learn
efficientnet
```

### Key Libraries Used:
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization
- **Scikit-learn**: Evaluation metrics

## 📁 Project Structure

```
deep-learning-mnist.ipynb  # Main Jupyter notebook with all experiments
README.md                  # This file
```

## 🎯 Future Improvements

1. **Data Augmentation**: Implement rotation, scaling, and noise addition
2. **Hyperparameter Tuning**: Optimize learning rates, batch sizes, and architectures
3. **Ensemble Methods**: Combine predictions from multiple models
4. **Advanced Architectures**: Try ResNet, DenseNet, or Vision Transformers
5. **Cross-validation**: Implement k-fold cross-validation for more robust evaluation

## 📚 References

- [MNIST Dataset Documentation](https://docs.rs/mnist/latest/mnist/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)

## 👨‍💻 Author

This project demonstrates various deep learning approaches for image classification, providing valuable insights into model selection and performance optimization for computer vision tasks.

---

**Note**: This project serves as an excellent learning resource for understanding the differences between various neural network architectures and their applications in image classification tasks. 