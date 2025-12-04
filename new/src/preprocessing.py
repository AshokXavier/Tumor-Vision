import tensorflow as tf

def gaussian_blur(image):
    """Applies a Gaussian blur to reduce noise while preserving edges better than avg_pool."""
    # Create a 2D Gaussian kernel
    kernel_size = 5
    sigma = 1.0
    ax = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
    xx, yy = tf.meshgrid(ax, ax)
    kernel = tf.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)
    kernel = tf.tile(kernel, [1, 1, 3, 1]) # Apply same filter to all 3 color channels

    # Apply the kernel using depthwise convolution
    blurred_image = tf.nn.depthwise_conv2d(
        tf.expand_dims(image, axis=0),
        kernel,
        strides=[1, 1, 1, 1],
        padding="SAME"
    )
    return tf.squeeze(blurred_image, axis=0)



def apply_preprocessing(image, label):
    """
    Applies a basic sequence of preprocessing steps to a single image.
    """
    # Step 1: Ensure consistent image size (resize to 128x128)
    image = tf.image.resize(image, [128, 128])
    
    # Step 2: Normalize pixel values from [0, 255] to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0

    # Step 3: Enhance Contrast to make features more distinct
    image = tf.image.adjust_contrast(image, contrast_factor=1.5)

    # Step 4: Apply brightness adjustment (final step)
    image = tf.image.adjust_brightness(image, delta=0.0)  # No-op that ensures proper format
    
    return image, label

def preprocess_dataset(ds):
    """
    Applies all preprocessing steps to an entire dataset.

    Args:
        ds (tf.data.Dataset): A dataset of (image, label) pairs.

    Returns:
        tf.data.Dataset: The fully preprocessed dataset.
    """
    return ds.map(apply_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)

