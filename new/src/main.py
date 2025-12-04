# Import functions from all other modules in the pipeline
from data_acquisition import load_and_split_datasets
from preprocessing import apply_preprocessing
from augmentation import apply_augmentation
from model_architecture import build_cnn_model
from training import train_model
from evaluation import evaluate_model
import json
import os
import tensorflow as tf

def main():
    """
    Main function to orchestrate the entire brain tumor classification pipeline.
    """
    print("Starting the Brain Tumor Classification pipeline...")

    # 1. Data Acquisition
    # Load the datasets and get the class names.
    print("\nStep 1: Loading and splitting datasets...")
    train_ds, val_ds, test_ds, class_names = load_and_split_datasets()
    num_classes = len(class_names)

    # Save the class names to a file for later use in prediction.
    with open("class_names.json", "w") as f:
        json.dump(class_names, f)
    print(f"Class names saved to class_names.json")

    # 2. Preprocessing
    # Apply the preprocessing steps to each dataset.
    print("\nStep 2: Applying preprocessing to datasets...")
    train_ds = train_ds.map(apply_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(apply_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(apply_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    print("Preprocessing complete.")

    # 3. Data Augmentation
    # Apply augmentation ONLY to the training dataset.
    print("\nStep 3: Applying data augmentation to the training dataset...")
    train_ds = apply_augmentation(train_ds)
    print("Augmentation complete.")

    # 4. Model Architecture
    # Build the CNN model.
    print("\nStep 4: Building the CNN model...")
    input_shape = (128, 128, 3)  # Input shape based on data acquisition settings
    model = build_cnn_model(input_shape=input_shape, num_classes=num_classes)
    model.summary()  # Print a summary of the model architecture.
    print("Model built successfully.")

    # 5. Model Training
    # Train the model using the prepared datasets.
    print("\nStep 5: Starting model training...")
    model_save_path = "best_brain_tumor_model_enhanced.keras"
    history = train_model(model, train_ds, val_ds, epochs=25, model_save_path=model_save_path)
    print("Training finished.")

    # 6. Model Evaluation
    # Evaluate the trained model on the unseen test dataset.
    print("\nStep 6: Evaluating the model on the test dataset...")
    report_path = "confusion_matrix.png"
    evaluate_model(model, test_ds, class_names, report_path)
    print("Evaluation complete.")

    print("\nPipeline finished successfully!")


if __name__ == "__main__":
    # This block ensures the main function runs only when the script is executed directly.
    main()

