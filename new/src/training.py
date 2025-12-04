import tensorflow as tf

def train_model(model, train_ds, val_ds, epochs, model_save_path):
    """
    Configures and trains the model.

    Args:
        model (keras.Model): The model to be trained.
        train_ds (tf.data.Dataset): The training data.
        val_ds (tf.data.Dataset): The validation data.
        epochs (int): The number of times to go through the entire training dataset.
        model_save_path (str): The file path where the best model will be saved.

    Returns:
        The training history object.
    """
    # Compile the model to configure the training process.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor="val_accuracy", restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_save_path, monitor="val_accuracy", save_best_only=True),
    ]

    # This is the main training loop.
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )
    return history
