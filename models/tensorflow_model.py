import tensorflow as tf
import numpy as np

# Simple Neural Network for classification (2 classes)
def create_simple_nn(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation='relu', input_shape=(input_dim,)), # hidden layer
        tf.keras.layers.Dense(1, activation='sigmoid') # output layer for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Example usage (uncomment to run):
# X = np.array([[0], [1], [2], [3], [4]])
# y = np.array([0, 0, 0, 1, 1])
# model = create_simple_nn(input_dim=1)
# model.fit(X, y, epochs=50, verbose=1)
# print('Predictions:', model.predict(X))
