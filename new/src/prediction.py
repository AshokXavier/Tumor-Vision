import numpy as np
from tensorflow import keras

def predict_single_image(model, class_names, image_path, img_size):
    """
    Loads an image, preprocesses it, and returns a prediction.

    Args:
        model (keras.Model): The trained model.
        class_names (list): A list of class names.
        image_path (str): The path to the image file.
        img_size (tuple): The target image size.

    Returns:
        tuple: A tuple of (predicted_label, confidence_score).
    """
    # Load the image from the file path.
    img = keras.utils.load_img(image_path, target_size=img_size)
    
    # Convert the image to a numpy array.
    img_array = keras.utils.img_to_array(img)
    
    # Add a batch dimension to create a batch of 1.
    img_array = np.expand_dims(img_array, axis=0)

    # The model expects normalized images, but the Rescaling layer is part of the dataset pipeline.
    # For a single image, we must normalize it manually.
    img_array /= 255.0

    # Make predictions.
    predictions = model.predict(img_array)
    
    # Determine the predicted class and its confidence.
    predicted_class_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    predicted_label = class_names[predicted_class_index]
    
    return predicted_label, confidence
