"""
This module implements a class that optimizes the weights of a Keras model 
using L-BFGS from TensorFlow Probability.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import keras
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


@dataclass
class LBfgsResultState:
    """
    Data class that retains the finish state of L-BFGS optimizer

    Attributes
    ----------
    converged: bool
        True if L-BFGS training converges
    failed: bool
        True if Wolf condition is satisfilled
    num_total_iterations: int
        Number of iteration when the training teminates
    loss_value: float
        Loss value when the training terminates
    """

    converged: bool
    failed: bool
    num_total_iterations: int
    loss_value: float


class LBfgsOptimizer:
    """
    Class to execute model training using L-BFGS in TensorFlow Probability

    Parameters
    ----------
    model: keras.models.Model
        Keras model to be optimized

    References
    ----------
    - https://www.tensorflow.org/probability/api_docs/python/tfp/optimizer/lbfgs_minimize
    """

    def __init__(
        self,
        model: keras.models.Model,
        loss_function: Callable,
        loss_function_kwargs: Dict[str, Any],
        lbfgs_kwargs: Dict[str, Any] = dict()
    ):
        self._model = model

        self._loss_function = loss_function

        self._loss_function_kwargs = loss_function_kwargs.copy()

        self._lbfgs_kwargs = lbfgs_kwargs.copy()

        stitch_indices, partition_indices = self._generate_weight_indices()
        self._stitch_indices = stitch_indices
        self._partition_indices = partition_indices

        self._previous_optimizer_results: Optional[
            tfp.optimizer.lbfgs.LBfgsOptimizerResults
        ] = None

    @property
    def weight_tensors(self) -> List:
        """
        Weighting of keras model
        """
        return self._model.trainable_weights

    @property
    def flatten_weight_array(self) -> tf.Tensor:
        """
        One-dimensional array of the keras model
        """
        return tf.dynamic_stitch(self._stitch_indices, self.weight_tensors)

    def _generate_weight_indices(self):
        """
        Calculate indices to convert the network weights of the keras model to the one-dimensional array.

        Returns
        -------
        stitch_indices: List[int]
            Indices to convert the network weigts into 1D array based on tf.dynamic_stitch
        partition_indices: List[int]
            Indices to covert 1D array into the same dimensional array as keras model based on tf.dynamic_partition

        """
        stitch_indices: List[int] = []
        partition_indices: List[int] = []
        cumulative_index = 0
        for i, variable in enumerate(self._model.trainable_weights):
            num_elements = np.product(variable.shape)
            stitch_indices.append(
                tf.reshape(
                    tf.range(
                        cumulative_index,
                        cumulative_index + num_elements,
                        dtype=tf.int32,
                    ),
                    variable.shape,
                )
            )
            partition_indices.extend([i] * num_elements)
            cumulative_index += num_elements

        return stitch_indices, partition_indices

    def _update_model_weights(self, flatten_new_weights_array: tf.Tensor):
        """
        Update the network weights of the keras model held by this object to the components of the passed one-dimensional array.

        Parameters
        ----------
        flatten_new_weights_array: tf.Tensor
            One-dimensial array that stores new network weights
        """
        weight_tensors = tf.dynamic_partition(
            flatten_new_weights_array,
            self._partition_indices,
            len(self._model.trainable_weights),
        )

        for i, weight_tensor in enumerate(weight_tensors):
            self._model.trainable_weights[i].assign(
                tf.reshape(weight_tensor, self._model.trainable_weights[i].shape)
            )
        return

    def optimize(self, max_iterations) -> LBfgsResultState:
        """
        Execute L-BFGS optmization

        Parameters
        ----------
        max_iterations: int
            Define maximum number of iteration

        Returns
        -------
        LBfgsResultState
            States of L-BFGS optmization when it terminates
        """
        # tfp.optimizer.lbfgs_optimize に渡す関数を生成します。
        def value_and_gradients_function(model_weight_array):
            """
            Function to pass to the value_and_gradients_function argument of tfp.optimizer.lbfgs_optimize.
            
            """
            self._update_model_weights(model_weight_array)

            loss_value, gradients = self._loss_function(**self._loss_function_kwargs)

            flatten_gradients_array = tf.dynamic_stitch(self._stitch_indices, gradients)

            # Return loss value and gradients
            return loss_value, flatten_gradients_array

        if self._previous_optimizer_results is None:
            results = tfp.optimizer.lbfgs_minimize(
                value_and_gradients_function,
                initial_position=self.flatten_weight_array,
                max_iterations=max_iterations,
                **self._lbfgs_kwargs
            )
        else:
            results = tfp.optimizer.lbfgs_minimize(
                value_and_gradients_function,
                initial_position=None,
                max_iterations=max_iterations,
                previous_optimizer_results=self._previous_optimizer_results,
                **self._lbfgs_kwargs
            )

        self._previous_optimizer_results = results

        self._update_model_weights(results.position)

        result_state = LBfgsResultState(
            converged=bool(results.converged),
            failed=bool(results.failed),
            num_total_iterations=int(results.num_iterations),
            loss_value=float(results.objective_value),
        )

        return result_state
