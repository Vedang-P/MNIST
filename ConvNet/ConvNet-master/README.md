# Pneumonia Detection using Convolutional Neural Networks

## Project Overview

This project implements a Convolutional Neural Network (CNN) to automatically detect pneumonia from chest X-ray images. The model achieves approximately 96% training accuracy and 93% test accuracy, making it a practical tool for assisting medical professionals in diagnosing pneumonia from radiographic images.

## What This Project Does

The CNN model analyzes chest X-ray images and classifies them into two categories:
- **Normal**: Clear lungs without abnormal opacification
- **Pneumonia**: Presence of pneumonia (both viral and bacterial types)

The model is designed to be computationally efficient while maintaining high accuracy, making it suitable for deployment in medical imaging workflows.

## Technical Architecture

### Model Structure
- **Input**: 128x128 pixel RGB chest X-ray images
- **Architecture**: Custom CNN with 7 convolutional blocks
- **Parameters**: 77,633 trainable parameters (much smaller than VGG16 or ResNet50)
- **Output**: Binary classification (pneumonia vs normal)

### Key Features
- **Spatial Dropout**: Prevents overfitting by randomly dropping entire feature maps
- **Batch Normalization**: Stabilizes training and improves convergence
- **Data Augmentation**: Horizontal flips, zoom, and shear transformations
- **Adam Optimizer**: Adaptive learning rate optimization

### Layer Breakdown
1. **Convolutional Layers**: 7 blocks of Conv2D + MaxPooling2D + SpatialDropout2D
2. **Flatten Layer**: Converts spatial features to 1D vector
3. **Dense Layers**: 3 fully connected layers with dropout regularization
4. **Output Layer**: Single neuron with sigmoid activation for binary classification

## Dataset

The model is trained on the Chest X-Ray Pneumonia dataset from Kaggle, which contains:
- **5,863 chest X-ray images**
- **Two classes**: Normal and Pneumonia
- **Quality control**: Images screened for readability by medical experts
- **Expert validation**: Diagnoses verified by two expert physicians

Dataset structure:
```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Setup and Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.4.0
- Keras 2.4.3
- Other dependencies listed in requirements.txt

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ConvNet-master
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Visit: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
   - Download and extract to the project directory
   - Ensure the folder structure matches the expected paths in CNN.py

## Usage

### Training the Model

1. **Prepare the dataset**
   - Place the chest X-ray images in the correct directory structure
   - Ensure images are organized in train/test splits

2. **Run the training script**
   ```bash
   python CNN.py
   ```

3. **Monitor training**
   - The model will train for 128 epochs
   - Training progress and validation metrics will be displayed
   - Model summary will be printed at the start

### Model Performance

- **Training Accuracy**: ~96%
- **Test Accuracy**: ~93%
- **Training Time**: Varies based on hardware (faster with GPU)

### Individual Image Prediction

To test the model on a single image, uncomment the section at the bottom of CNN.py and modify the image path:

```python
test_image = image.load_img('path/to/your/image.jpg', target_size=(128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict_classes(test_image)
print(result)
```

## Model Adaptability

This CNN can be adapted for other image classification tasks:

### Binary Classification
- Replace the dataset with your own two-class images
- Keep the existing architecture unchanged

### Multi-Class Classification
For more than 2 classes, modify the final layer:

```python
# Change the final dense layer
model.add(Dense(num_classes, activation='softmax'))

# Update the compilation
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
```

## Technical Details

### Data Preprocessing
- **Rescaling**: Images normalized to [0,1] range
- **Resizing**: All images resized to 128x128 pixels
- **Augmentation**: Random horizontal flips, zoom, and shear for training

### Training Configuration
- **Batch Size**: 16
- **Learning Rate**: 0.001
- **Loss Function**: Binary cross-entropy
- **Optimizer**: Adam
- **Validation**: 20% of training data

### Regularization Techniques
- **Spatial Dropout**: 0.1-0.5 rates in convolutional layers
- **Dropout**: 0.5 rate in dense layers
- **Batch Normalization**: Applied after convolutional layers

## Limitations

- **Binary Classification**: Does not distinguish between viral and bacterial pneumonia
- **Image Quality**: Performance depends on image quality and preprocessing
- **Dataset Bias**: Model performance is specific to the training dataset characteristics

## Medical Disclaimer

This model is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## Contributing

Contributions to improve the model architecture, add new features, or enhance documentation are welcome. Please ensure any changes maintain the model's medical accuracy and reliability.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Dataset provided by Paul Mooney on Kaggle
- Medical expertise validation by expert physicians
- Keras and TensorFlow communities for the deep learning framework
