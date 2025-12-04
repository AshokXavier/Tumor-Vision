import tensorflow as tf
import os

# Define constants for image size and batch size for consistency.
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

def load_and_split_datasets(base_dir="data/dataset"):
    """
    Loads the dataset from the base directory, splitting the training data
    into training and validation sets.

    Args:
        base_dir (str): The path to the root dataset directory which contains
                        'Training' and 'Testing' subfolders.

    Returns:
        A tuple containing:
        - train_ds (tf.data.Dataset): The training dataset.
        - val_ds (tf.data.Dataset): The validation dataset.
        - test_ds (tf.data.Dataset): The testing dataset.
        - class_names (list): A list of the class names found.
    """
    train_dir = os.path.join(base_dir, "Training")
    test_dir = os.path.join(base_dir, "Testing")

    # Load the training data and split it into training (80%) and validation (20%) sets.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"  # Use integer labels for sparse categorical crossentropy
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    # Load the testing dataset from its separate directory.
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    class_names = train_ds.class_names
    print(f"Found class names: {class_names}")

    # Configure dataset for performance by caching and prefetching.
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

