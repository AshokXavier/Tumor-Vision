import tensorflow as tf
from tensorflow import keras

def apply_augmentation(ds):
    """
    Applies random transformations to the training dataset.

    Data augmentation creates modified versions of the training images, which helps
    the model generalize better to new, unseen images.

    Args:
        ds (tf.data.Dataset): The training dataset.

    Returns:
        tf.data.Dataset: The augmented training dataset.
    """
    # Create a sequence of augmentation layers.
    data_augmentation = keras.Sequential([
        # Randomly flip images horizontally.
        keras.layers.RandomFlip("horizontal"),
        # Randomly rotate images by a small amount.
        keras.layers.RandomRotation(0.1),
        # Randomly shift images horizontally or vertically.
        keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    ])

    # Apply the augmentation layers to the training dataset.
    # This is only done during 'training' so the model sees varied images.
    return ds.map(
        lambda image, label: (data_augmentation(image, training=True), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )
