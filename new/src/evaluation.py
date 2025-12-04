import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf

def evaluate_model(model, test_ds, class_names, report_path):
    """
    Evaluates the model on the test dataset and saves a confusion matrix.

    Args:
        model (keras.Model): The trained model.
        test_ds (tf.data.Dataset): The dataset for testing.
        class_names (list): The list of class names for labeling the report.
        report_path (str): The path to save the confusion matrix image.
    """
    # Get the model's predictions for the entire test set.
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Get the true labels from the test set.
    y_true = np.concatenate([y for x, y in test_ds], axis=0)

    # Calculate accuracy.
    acc = accuracy_score(y_true, y_pred)
    # The classification report provides precision, recall, and f1-score for each class.
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    print("\n--- Model Evaluation Results ---")
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)

    # The confusion matrix shows where the model got confused (e.g., predicted 'glioma' when it was 'meningioma').
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(report_path)
    print(f"\nConfusion matrix saved to {report_path}")
    plt.close()

def display_accuracy_computation(model, test_ds, class_names):
    """
    Displays detailed accuracy computation with step-by-step explanation.
    
    Args:
        model (keras.Model): The trained model.
        test_ds (tf.data.Dataset): The test dataset.
        class_names (list): The list of class names.
    """
    print("\n" + "="*70)
    print("🎯 DETAILED ACCURACY COMPUTATION EXPLANATION")
    print("="*70)
    
    # Get predictions and true labels
    print("\n📊 Step 1: Getting Model Predictions")
    print("-" * 40)
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    
    total_samples = len(y_true)
    print(f"Total test samples: {total_samples}")
    print(f"Prediction probabilities shape: {y_pred_probs.shape}")
    print(f"Classes: {class_names}")
    
    # Show some example predictions
    print("\n🔍 Example Predictions (First 5 samples):")
    for i in range(min(5, total_samples)):
        true_class = class_names[y_true[i]]
        pred_class = class_names[y_pred[i]]
        confidence = y_pred_probs[i, y_pred[i]] * 100
        correct = "✅" if y_true[i] == y_pred[i] else "❌"
        print(f"Sample {i+1}: True={true_class}, Predicted={pred_class}, Confidence={confidence:.1f}% {correct}")
    
    # Calculate accuracy step by step
    print("\n🧮 Step 2: Accuracy Calculation")
    print("-" * 40)
    print("Accuracy Formula: (Number of Correct Predictions) / (Total Number of Predictions)")
    
    correct_predictions = np.sum(y_true == y_pred)
    accuracy_manual = correct_predictions / total_samples
    accuracy_sklearn = accuracy_score(y_true, y_pred)
    
    print(f"\nCorrect predictions: {correct_predictions}")
    print(f"Total predictions: {total_samples}")
    print(f"Manual calculation: {correct_predictions}/{total_samples} = {accuracy_manual:.4f}")
    print(f"Using sklearn: {accuracy_sklearn:.4f}")
    print(f"Accuracy percentage: {accuracy_manual * 100:.2f}%")
    
    # Per-class accuracy breakdown
    print("\n📈 Step 3: Per-Class Accuracy Breakdown")
    print("-" * 40)
    
    for i, class_name in enumerate(class_names):
        # Get samples for this class
        class_mask = (y_true == i)
        class_samples = np.sum(class_mask)
        
        if class_samples > 0:
            class_correct = np.sum((y_true == i) & (y_pred == i))
            class_accuracy = class_correct / class_samples
            print(f"{class_name:12}: {class_correct:3d}/{class_samples:3d} correct = {class_accuracy:.3f} ({class_accuracy*100:.1f}%)")
        else:
            print(f"{class_name:12}: No samples in test set")
    
    # Show confusion matrix breakdown
    print("\n🔥 Step 4: Confusion Matrix Analysis")
    print("-" * 40)
    cm = confusion_matrix(y_true, y_pred)
    
    print("Confusion Matrix:")
    print("Rows = True Labels, Columns = Predicted Labels")
    print("\n" + " " * 12 + "".join([f"{name:>12}" for name in class_names]))
    for i, true_name in enumerate(class_names):
        row_str = f"{true_name:>12}"
        for j in range(len(class_names)):
            row_str += f"{cm[i,j]:>12}"
        print(row_str)
    
    # Diagonal elements are correct predictions
    print(f"\nDiagonal sum (correct predictions): {np.trace(cm)}")
    print(f"Total predictions: {np.sum(cm)}")
    print(f"Accuracy from confusion matrix: {np.trace(cm)/np.sum(cm):.4f}")
    
    # Common mistakes analysis
    print("\n❌ Step 5: Common Classification Mistakes")
    print("-" * 40)
    
    mistakes = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i,j] > 0:
                mistakes.append((cm[i,j], class_names[i], class_names[j]))
    
    mistakes.sort(reverse=True)
    print("Most common misclassifications:")
    for count, true_class, pred_class in mistakes[:5]:
        if count > 0:
            print(f"  {true_class} → {pred_class}: {count} times")
    
    print("\n" + "="*70)
    print(f"🎯 FINAL ACCURACY: {accuracy_manual:.4f} ({accuracy_manual*100:.2f}%)")
    print("="*70)
    
    return accuracy_manual, correct_predictions, total_samples
