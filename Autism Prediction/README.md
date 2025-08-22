# Autism Spectrum Disorder (ASD) Prediction

A machine learning project that predicts Autism Spectrum Disorder (ASD) based on behavioral assessment scores and demographic information. This project implements a complete machine learning pipeline from data preprocessing to model deployment.

## Project Overview

This project uses machine learning algorithms to classify individuals as having ASD or not based on screening data. The model can serve as a preliminary screening tool to identify individuals who may need further clinical evaluation for autism spectrum disorder.

## Dataset Description

The dataset contains 800 samples with 22 features including:

- **Behavioral Assessment Scores (A1-A10)**: 10 binary scores from autism screening questionnaires
- **Demographic Information**: Age, gender, ethnicity, country of residence
- **Medical History**: Jaundice at birth, family history of autism
- **Screening Context**: Whether the app was used before, relationship to the person being screened
- **Target Variable**: Class/ASD (0 = No ASD, 1 = ASD)

## Features

- Comprehensive data preprocessing and cleaning
- Exploratory data analysis with visualizations
- Class imbalance handling using SMOTE
- Multiple machine learning algorithms comparison
- Hyperparameter optimization using RandomizedSearchCV
- Model evaluation with detailed metrics
- Model persistence for deployment

## Requirements

### Python Version
- Python 3.7 or higher

### Dependencies
```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
pickle
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd autism-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
autism-prediction/
├── Autism Prediction.ipynb    # Main Jupyter notebook
├── train.csv                  # Dataset file
├── encoders.pkl              # Saved label encoders
├── best_model.pkl            # Trained model
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Usage

### Running the Notebook

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `Autism Prediction.ipynb`

3. Run all cells sequentially to execute the complete pipeline

### Making Predictions

To use the trained model for new predictions:

```python
import pickle
import pandas as pd

# Load the model and encoders
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Prepare new data (ensure it has the same features as training data)
# Apply the same preprocessing steps
# Make prediction
prediction = model.predict(new_data)
```

## Methodology

### Data Preprocessing

1. **Data Cleaning**:
   - Removed unnecessary columns (ID, age_desc)
   - Standardized country names
   - Cleaned ethnicity and relationship categories
   - Converted age to integer type

2. **Feature Engineering**:
   - Applied label encoding to categorical variables
   - Handled outliers using IQR method with median replacement

3. **Data Splitting**:
   - 80% training set, 20% test set
   - Applied SMOTE to handle class imbalance

### Model Development

Three classification algorithms were evaluated:

1. **Decision Tree Classifier**
2. **Random Forest Classifier**
3. **XGBoost Classifier**

### Hyperparameter Optimization

Used RandomizedSearchCV with the following parameter grids:

- **Decision Tree**: criterion, max_depth, min_samples_split, min_samples_leaf
- **Random Forest**: n_estimators, max_depth, min_samples_split, min_samples_leaf, bootstrap
- **XGBoost**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree

## Results

### Model Performance

| Model | Cross-Validation Accuracy | Test Accuracy |
|-------|---------------------------|---------------|
| Decision Tree | 86% | - |
| Random Forest | 92% | 82% |
| XGBoost | 90% | - |

### Final Model Metrics (Random Forest)

- **Overall Accuracy**: 82%
- **Precision**: 89% (non-ASD), 59% (ASD)
- **Recall**: 87% (non-ASD), 64% (ASD)
- **F1-Score**: 88% (non-ASD), 61% (ASD)

## Key Findings

1. **Class Imbalance**: The dataset contains more non-ASD cases than ASD cases, which is typical for medical screening datasets.

2. **Best Performing Model**: Random Forest achieved the highest cross-validation accuracy (92%) and was selected as the final model.

3. **Feature Importance**: The A1-A10 behavioral assessment scores are likely the most important features for prediction.

4. **Model Limitations**: The model shows lower performance on ASD cases (class 1) compared to non-ASD cases, which is common in imbalanced medical datasets.

## Data Visualization

The project includes comprehensive visualizations:

- Age and result score distributions
- Box plots for outlier detection
- Count plots for categorical variables
- Correlation heatmap for feature relationships

## Model Deployment

The trained model and preprocessing encoders are saved as pickle files:

- `best_model.pkl`: The optimized Random Forest classifier
- `encoders.pkl`: Label encoders for categorical variables

These files can be used to make predictions on new data without retraining the model.

## Limitations and Considerations

1. **Medical Disclaimer**: This model is for screening purposes only and should not replace professional medical diagnosis.

2. **Dataset Limitations**: The model's performance may vary with different populations or screening tools.

3. **Feature Dependencies**: The model requires all 20 features to be present for accurate predictions.

4. **Class Imbalance**: The model performs better on non-ASD cases due to the imbalanced nature of the dataset.

## Future Improvements

1. **Data Collection**: Gather more balanced datasets with equal representation of ASD and non-ASD cases.

2. **Feature Engineering**: Explore additional features or feature combinations that might improve prediction accuracy.

3. **Model Ensemble**: Combine multiple models to improve overall performance.

4. **Real-time Application**: Develop a web application for easy access and use.

5. **Validation Studies**: Conduct validation studies with different populations and screening tools.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please open an issue in the repository.

## Acknowledgments

- Dataset providers for making the autism screening data available
- Open-source machine learning community for the tools and libraries used
- Medical professionals who provided domain expertise for this project 