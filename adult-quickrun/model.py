import tensorflow as tf

from declearn.model.tensorflow import TensorflowModel

stack = [
    tf.keras.layers.InputLayer(shape=[11]),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),  # Binary classification
]
network = tf.keras.models.Sequential(stack)

# This needs to be called "model"; otherwise, a different name must be
# specified via the experiment's TOML configuration file.
model = TensorflowModel(network, loss="binary_crossentropy")
