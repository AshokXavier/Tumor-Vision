#!/usr/bin/env python3
"""
Script to display detailed accuracy computation for the brain tumor classification model.
"""

import tensorflow as tf
import json
import sys
import os

# Add src folder to path
sys.path.append('src')

from evaluation import display_accuracy_computation
from data_acquisition import load_and_split_datasets
from preprocessing import preprocess_dataset

def main():
    """Display accuracy computation details."""
    print("🧠 Brain Tumor Classification - Accuracy Computation Display")
    print("="*70)
    
    # Load the trained model
    model_path = "best_brain_tumor_model_enhanced.keras"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first by running: python src/main.py")
        return
    
    print(f"📁 Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Load class names
    class_names_path = "class_names.json"
    if not os.path.exists(class_names_path):
        print(f"❌ Class names not found at {class_names_path}")
        return
    
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    print(f"📋 Classes: {class_names}")
    
    # Create test dataset
    print(f"\n📊 Creating test dataset...")
    data_dir = "data/dataset"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found at {data_dir}")
        return
    
    # Create datasets (they're already batched from load_and_split_datasets)
    train_ds, val_ds, test_ds, _ = load_and_split_datasets(data_dir)
    
    # Apply preprocessing to test dataset (unbatch first, then rebatch)
    test_ds = test_ds.unbatch()
    test_ds = preprocess_dataset(test_ds)
    test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)
    
    # Display detailed accuracy computation
    print(f"\n🎯 Analyzing model performance on test dataset...")
    accuracy, correct, total = display_accuracy_computation(model, test_ds, class_names)
    
    print(f"\n✅ Analysis complete!")
    print(f"Final Summary: {correct}/{total} correct predictions = {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    main()