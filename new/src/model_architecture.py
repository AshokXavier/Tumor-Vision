from tensorflow import keras

def build_cnn_model(input_shape, num_classes):
    """
    Builds and returns the Convolutional Neural Network (CNN) model.

    Args:
        input_shape (tuple): The shape of the input images (e.g., (128, 128, 3)).
        num_classes (int): The number of possible output categories (e.g., 4 for 4 tumor types).

    Returns:
        A Keras Sequential model.
    """
    model = keras.models.Sequential([
        # This layer specifies the input shape for the model.
        keras.layers.Input(shape=input_shape),

        # First Convolutional Block: Learns basic features like edges and textures.
        # Conv2D applies 32 filters to find patterns. 'relu' is an activation function that helps the model learn non-linear patterns.
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        # MaxPooling2D reduces the size of the image, keeping the most important information.
        keras.layers.MaxPooling2D(),

        # Second Convolutional Block: Learns more complex features from the previous block.
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D(),

        # Third Convolutional Block: Learns even more complex patterns.
        keras.layers.Conv2D(128, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D(),

        # Classifier Head: Takes the learned features and makes a final decision.
        # Flatten converts the 2D feature maps into a 1D vector.
        keras.layers.Flatten(),
        # Dense is a standard fully-connected layer for classification.
        keras.layers.Dense(128, activation="relu"),
        # Dropout randomly deactivates some neurons during training to prevent overfitting.
        keras.layers.Dropout(0.5),
        # Final Dense layer with 'softmax' activation. Softmax outputs a probability score for each class.
        keras.layers.Dense(num_classes, activation="softmax")
    ])

    return model
