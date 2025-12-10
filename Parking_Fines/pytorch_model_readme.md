# PyTorch Neural Network Models for NYC Parking Fines Prediction

This directory contains PyTorch-based deep learning models for predicting NYC parking fines, implementing both regression and classification tasks using neural networks.

## 🎯 Overview

The PyTorch implementation provides two neural network models:

1. **Regression Model (`FineAmountRegressor`)**: Predicts the exact parking fine amount in dollars
2. **Classification Model (`FineCategoryClassifier`)**: Predicts the fine category (small, medium, or large)

These models complement the existing machine learning approaches (Linear Regression, Random Forest, XGBoost) by leveraging deep neural networks to capture complex non-linear patterns in the data.

## 🏗️ Architecture

### Regression Model Architecture

```
Input Layer (variable input_dim based on preprocessed features)
    ↓
Dense Layer (256 neurons) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense Layer (128 neurons) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense Layer (64 neurons) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Output Layer (1 neuron for regression)
```

### Classification Model Architecture

```
Input Layer (variable input_dim based on preprocessed features)
    ↓
Dense Layer (256 neurons) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense Layer (128 neurons) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense Layer (64 neurons) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Output Layer (3 neurons for 3 categories: small, medium, large)
```

### Key Features

- **Batch Normalization**: Stabilizes training and accelerates convergence
- **Dropout Regularization**: Prevents overfitting (30% dropout rate)
- **ReLU Activation**: Introduces non-linearity
- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Adam Optimizer**: Adaptive learning rate optimization

## 📦 Installation

### Prerequisites

Ensure you have Python 3.8+ installed.

### Install Dependencies

```bash
cd Parking_Fines
pip install -r requirements_pytorch.txt
```

This will install:
- PyTorch (GPU-accelerated if CUDA is available)
- NumPy and Pandas for data processing
- scikit-learn for preprocessing and metrics
- Matplotlib and Seaborn for visualization

## 🚀 Usage

### Training Models

To train both regression and classification models:

```bash
cd Parking_Fines
python train_pytorch_model.py
```

The script will:
1. Load and preprocess the parking violations dataset
2. Split data into training and testing sets (70/30 split)
3. Train both models with early stopping
4. Save trained models to `models/` directory
5. Display evaluation metrics

**Note**: The training script assumes data files are in the `data/` directory:
- `data/Parking_Violations_Issued_-_Fiscal_Year_2015.csv`
- `data/ParkingViolationCodes.csv`

### Using Pre-trained Models

To load and use trained models for prediction:

```python
import torch
import pickle
from pytorch_model import FineAmountRegressor, FineCategoryClassifier

# Load preprocessor
with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Load regression model
checkpoint = torch.load('models/pytorch_regressor.pth')
reg_model = FineAmountRegressor(input_dim=checkpoint['input_dim'])
reg_model.load_state_dict(checkpoint['model_state_dict'])
reg_model.eval()

# Load classification model
checkpoint = torch.load('models/pytorch_classifier.pth')
class_model = FineCategoryClassifier(
    input_dim=checkpoint['input_dim'],
    num_classes=checkpoint['num_classes']
)
class_model.load_state_dict(checkpoint['model_state_dict'])
class_model.eval()

# Preprocess new data
X_new_processed = preprocessor.transform(X_new)

# Make predictions
with torch.no_grad():
    fine_predictions = reg_model(torch.FloatTensor(X_new_processed))
    category_predictions = class_model(torch.FloatTensor(X_new_processed))
```

## 📊 Performance Metrics

The models are evaluated using standard metrics:

### Regression Model Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual fine amounts
- **RMSE (Root Mean Squared Error)**: Square root of average squared differences
- **R² Score**: Proportion of variance explained by the model

### Classification Model Metrics
- **Accuracy**: Percentage of correctly classified fine categories
- **F1 Score (Macro)**: Harmonic mean of precision and recall, averaged across all classes
- **Classification Report**: Detailed per-class precision, recall, and F1 scores

### Comparison with Existing Models

After training, you can compare PyTorch models with the existing models in `nyc_fines.ipynb`:

| Model | Task | MAE | RMSE | R² | Accuracy | F1 (Macro) |
|-------|------|-----|------|----|---------:|------------|
| Linear Regression | Regression | ~X | ~X | ~X | - | - |
| Random Forest | Regression | ~X | ~X | ~X | - | - |
| XGBoost | Regression | ~X | ~X | ~X | - | - |
| **PyTorch NN** | **Regression** | **~X** | **~X** | **~X** | - | - |
| XGBoost | Classification | - | - | - | ~X | ~X |
| **PyTorch NN** | **Classification** | - | - | - | **~X** | **~X** |

*Note: Run the training script to get actual performance metrics.*

## 🔧 Customization

### Hyperparameter Tuning

You can customize the model architecture and training process:

```python
# Custom architecture
model = FineAmountRegressor(
    input_dim=input_dim,
    hidden_dims=[512, 256, 128, 64],  # More/different layers
    dropout_rate=0.4  # Higher dropout
)

# Custom training parameters
trainer.train(
    train_loader=train_loader,
    val_loader=test_loader,
    optimizer=optimizer,
    criterion=criterion,
    num_epochs=100,  # More epochs
    early_stopping_patience=15,  # More patience
    verbose=True
)
```

### Learning Rate Scheduling

Add a learning rate scheduler for better convergence:

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
```

## 📁 File Structure

```
Parking_Fines/
├── pytorch_model.py              # Model architectures and utilities
├── train_pytorch_model.py        # Training script
├── requirements_pytorch.txt      # PyTorch dependencies
├── pytorch_model_readme.md       # This file
├── models/                       # Saved models (created after training)
│   ├── pytorch_regressor.pth     # Trained regression model
│   ├── pytorch_classifier.pth    # Trained classification model
│   └── preprocessor.pkl          # Feature preprocessor
└── data/                         # Data files (not in repository)
    ├── Parking_Violations_Issued_-_Fiscal_Year_2015.csv
    └── ParkingViolationCodes.csv
```

## 🧠 Model Training Details

### Data Preprocessing

1. **Feature Engineering**:
   - Cyclical encoding of violation hour (sine/cosine transformation)
   - Date component extraction (year, month, day)
   - Fine category binning: small ($0-50), medium ($50-100), large ($100+)

2. **Feature Preprocessing**:
   - **Numerical features**: StandardScaler (without mean centering for sparse data)
   - **Categorical features**: OneHotEncoder
   - Missing values are dropped before training

3. **Data Split**: 70% training, 30% testing (random_state=42 for reproducibility)

### Training Configuration

- **Batch Size**: 512 (efficient GPU utilization)
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-5 for L2 regularization)
- **Loss Functions**:
  - Regression: MSE (Mean Squared Error)
  - Classification: CrossEntropyLoss
- **Early Stopping**: Monitors validation loss with patience of 10 epochs
- **Device**: Automatically uses CUDA if available, otherwise CPU

### Training Tips

1. **GPU Acceleration**: If you have a CUDA-capable GPU, PyTorch will automatically use it for faster training
2. **Memory Management**: If you encounter out-of-memory errors, reduce batch size or sample size
3. **Convergence**: Models typically converge within 20-30 epochs with early stopping

## 🔍 Advantages of Neural Networks

### Why PyTorch Neural Networks?

1. **Complex Pattern Recognition**: Neural networks can capture non-linear relationships that traditional ML models might miss
2. **Feature Learning**: Automatically learns relevant feature combinations
3. **Scalability**: Efficient GPU acceleration for large datasets
4. **Flexibility**: Easy to customize architecture for specific needs
5. **Transfer Learning**: Pre-trained layers can be reused for related tasks

### When to Use Neural Networks vs. Traditional ML

**Use Neural Networks when**:
- Dataset is large (>100k samples)
- Complex non-linear patterns exist
- GPU resources are available
- High accuracy is critical

**Use Traditional ML (XGBoost, Random Forest) when**:
- Dataset is small (<10k samples)
- Interpretability is important
- Training time is limited
- Resource constraints exist

## 📈 Future Improvements

Potential enhancements for the models:

1. **Hyperparameter Optimization**: Use tools like Optuna or Ray Tune
2. **Cross-Validation**: Implement k-fold cross-validation for robust evaluation
3. **Ensemble Methods**: Combine PyTorch models with XGBoost for better predictions
4. **Advanced Architectures**: 
   - Attention mechanisms for feature importance
   - Residual connections for deeper networks
5. **Feature Engineering**: 
   - Geographic embeddings for location data
   - Temporal patterns (seasonality, trends)
6. **Model Deployment**: Create REST API for real-time predictions

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [NYC OpenData - Parking Violations](https://data.cityofnewyork.us/City-Government/Parking-Violations-Issued-Fiscal-Year-2023/869v-vr48)
- [Neural Network Best Practices](https://pytorch.org/tutorials/beginner/nn_tutorial.html)

## 🤝 Contributing

This implementation follows the same data preprocessing pipeline as the original notebook to ensure fair comparison with existing models.

## 📝 License

This project follows the same license as the main NYC-ML-Project repository.
