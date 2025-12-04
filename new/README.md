# Brain Tumor Classification Project 🧠

A deep learning project for classifying brain tumors from MRI images using TensorFlow/Keras.

## 🎯 Project Overview

This project implements a Convolutional Neural Network (CNN) to classify brain MRI images into four categories:
- **Glioma** - A type of brain tumor
- **Meningioma** - A tumor of the brain's protective membranes
- **No Tumor** - Healthy brain tissue
- **Pituitary** - Pituitary gland tumor

## 📊 Model Performance

- **Test Accuracy**: 86.58% ✨
- **Model Size**: 16.6 MB
- **Parameters**: 1.44M
- **Training Time**: ~20 minutes (25 epochs with early stopping)

### Per-Class Performance:
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 93.05% | 80.33% | 86.23% |
| Meningioma | 80.43% | 72.55% | 76.29% |
| No Tumor | 93.10% | 93.33% | 93.22% |
| Pituitary | 79.46% | 98.00% | 87.76% |

## 🏗️ Architecture

Enhanced CNN with:
- 4 Convolutional blocks with Batch Normalization
- Dropout layers for regularization
- Global Average Pooling
- Dense layers with 512→256→4 neurons
- Advanced preprocessing pipeline

## 📁 Project Structure

```
brain-tumor-classification/
├── src/
│   ├── main.py                 # Main training pipeline
│   ├── data_acquisition.py     # Data loading and splitting
│   ├── preprocessing.py        # Image preprocessing
│   ├── augmentation.py         # Data augmentation
│   ├── model_architecture.py   # CNN model definition
│   ├── training.py            # Training configuration
│   ├── evaluation.py          # Model evaluation
│   ├── prediction.py          # Single image prediction
│   └── interactive_prediction.py # Interactive testing
├── data/
│   └── dataset/
│       ├── Training/          # Training images
│       └── Testing/           # Test images
├── app.py                     # Complete application
├── demo.py                    # Quick demo script
├── best_brain_tumor_model_enhanced.keras # Trained model
├── class_names.json           # Class labels
├── confusion_matrix.png       # Performance visualization
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install tensorflow numpy matplotlib scikit-learn Pillow seaborn
```

### 2. Train Model (Optional - Model Already Trained)
```bash
python src/main.py
```

### 3. Test the Model

#### Quick Demo:
```bash
python demo.py
```

#### Comprehensive Application:
```bash
# Demo mode (3 samples per class)
python app.py --mode demo

# Batch testing (5 samples per class)
python app.py --mode batch --samples 5

# Interactive mode
python app.py --mode interactive
```

#### Manual Testing:
```bash
python src/interactive_prediction.py
```

## 📈 Training Process

The model was trained with:
- **Enhanced preprocessing**: Standardization + contrast/saturation adjustment
- **Data augmentation**: Rotations, flips, zoom, shifts
- **Advanced architecture**: Batch normalization + dropout
- **Smart training**: Learning rate scheduling + early stopping
- **Validation**: 80/20 train/validation split

Training progress:
- Started at ~55% accuracy
- Reached 87.81% validation accuracy at epoch 11
- Early stopping prevented overfitting
- Learning rate reduced automatically when plateauing

## 🔬 Key Features

### Preprocessing Pipeline:
1. **Normalization**: [0, 255] → [0, 1]
2. **Standardization**: Per-image mean/std normalization
3. **Contrast Enhancement**: 1.2x contrast boost
4. **Saturation Boost**: 1.1x saturation increase

### Model Architecture:
- **Input**: 128×128×3 RGB images
- **4 Conv Blocks**: 32→64→128→256 filters
- **Regularization**: Batch norm + dropout (0.25, 0.5)
- **Pooling**: Global Average Pooling (reduces overfitting)
- **Classification**: 512→256→4 dense layers

### Training Features:
- **Optimizer**: Adam (learning rate: 5e-4)
- **Loss**: Sparse categorical crossentropy
- **Callbacks**: Early stopping, model checkpointing, LR scheduling
- **Epochs**: 25 (stopped early at 18)

## 📊 Results Analysis

### Confusion Matrix:
View `confusion_matrix.png` for detailed per-class performance.

### Strong Points:
- Excellent at detecting "No Tumor" cases (93.33% recall)
- High precision for Glioma detection (93.05%)
- Good overall balance across classes

### Areas for Improvement:
- Meningioma classification could be enhanced
- Some confusion between Glioma and other tumor types

## 🛠️ Usage Examples

### Single Image Prediction:
```python
from app import BrainTumorClassifier

classifier = BrainTumorClassifier()
classifier.load_model()

result = classifier.predict_single_image("path/to/image.jpg")
print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### Batch Processing:
```python
results, true_labels, predictions = classifier.batch_predict(
    "data/dataset/Testing", num_samples_per_class=10
)
accuracy, class_stats = classifier.analyze_results(results)
```

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Scikit-learn
- Pillow
- Seaborn

## 🎓 Medical Imaging Context

This model achieves excellent performance for medical image classification:
- **86.58% accuracy** is very good for brain tumor classification
- Professional medical systems typically achieve 70-90% accuracy
- The model shows good generalization across different tumor types
- High recall for "No Tumor" cases is medically important

## 🚀 Next Steps

1. **Deploy the model** in a web application
2. **Collect more data** to improve Meningioma classification
3. **Implement ensemble methods** for higher accuracy
4. **Add uncertainty quantification** for medical reliability
5. **Create mobile app** for point-of-care diagnosis

## 📜 License

This project is for educational and research purposes. For medical applications, proper validation and regulatory approval would be required.

## 🤝 Contributing

Feel free to contribute by:
- Adding more preprocessing techniques
- Implementing different architectures
- Improving the web interface
- Adding more evaluation metrics

---

**Disclaimer**: This model is for educational purposes only and should not be used for actual medical diagnosis without proper validation and approval from medical professionals.
