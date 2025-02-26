# GRADED FUNCTION: train_model
import numpy as np
import tensorflow as tf
import test1
def train_model():
    """Returns the trained model.

    Returns:
        tf.keras.Model: The trained model that will predict house prices.
    """

    ### START CODE HERE ###

    # Define feature and target tensors with the values for houses with 1 up to 6 bedrooms
    # Hint: Remember you already coded a function that does this!
    n_bedrooms = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    price_in_hundreds_of_thousands = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

    # Define a compiled (but untrained) model
    # Hint: Remember you already coded a function that does this!
    model = tf.keras.Sequential([
        # Define the Input with the appropriate shape
        tf.keras.Input(shape=(1,)),
        # Define the Dense layer
        tf.keras.layers.Dense(units=1)
    ])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    # Train your model for 500 epochs by feeding the training data
    model.fit(n_bedrooms, price_in_hundreds_of_thousands, epochs=500)

    ### END CODE HERE ###

    return model