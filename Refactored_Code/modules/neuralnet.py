"""
Module for neural network architecture
"""

import tensorflow as tf


# Define PINN model architecture
class PINN_NeuralNet3(tf.keras.Model):
    """Set architecture of the PINN model."""

    def __init__(
        self,
        lb,
        ub,
        output_dim=1,
        num_hidden_layers=4,
        num_neurons_per_layer=50,
        activation="tanh",
        kernel_initializer="glorot_normal",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        self.lb = lb
        self.ub = ub

        # Define NN architecture
        self.scale1 = tf.keras.layers.Lambda(lambda x: 2.0 * (x - lb) / (ub - lb) - 1.0)
        self.hidden1 = [
            tf.keras.layers.Dense(
                num_neurons_per_layer,
                activation=tf.keras.activations.get(activation),
                kernel_initializer=kernel_initializer,
            )
            for _ in range(self.num_hidden_layers)
        ]
        self.out1 = tf.keras.layers.Dense(output_dim, activation="sigmoid")
        # self.out1 = tf.keras.layers.Dense(output_dim)

        self.scale2 = tf.keras.layers.Lambda(lambda x: 2.0 * (x - lb) / (ub - lb) - 1.0)
        self.hidden2 = [
            tf.keras.layers.Dense(
                num_neurons_per_layer,
                activation=tf.keras.activations.get(activation),
                kernel_initializer=kernel_initializer,
            )
            for _ in range(self.num_hidden_layers)
        ]
        self.out2 = tf.keras.layers.Dense(output_dim, activation="sigmoid")
        # self.out2 = tf.keras.layers.Dense(output_dim)

        self.scale3 = tf.keras.layers.Lambda(lambda x: 2.0 * (x - lb) / (ub - lb) - 1.0)
        self.hidden3 = [
            tf.keras.layers.Dense(
                num_neurons_per_layer,
                activation=tf.keras.activations.get(activation),
                kernel_initializer=kernel_initializer,
            )
            for _ in range(self.num_hidden_layers)
        ]
        self.out3 = tf.keras.layers.Dense(output_dim, activation="sigmoid")
        # self.out3 = tf.keras.layers.Dense(output_dim)

    def call(self, X):
        """Forward-pass through neural network."""
        Z1 = self.scale1(X)
        Z2 = self.scale2(X)
        Z3 = self.scale3(X)
        for i in range(self.num_hidden_layers):
            Z1 = self.hidden1[i](Z1)
            Z2 = self.hidden2[i](Z2)
            Z3 = self.hidden3[i](Z3)
        return self.out1(Z1), self.out2(Z2), self.out3(Z3)
