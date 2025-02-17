from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import keras
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


@dataclass
class LBfgsResultState:
    """
    Data class that holds information about the state at the end of L-BFGS

    Attributes
    ----------
    converged: bool
        This will be true if the convergence of L-BFGS has been reached.
    failed: bool
        This will be true if the linear search fails to satisfy the Wolfe condition.
    num_total_iterations: int
        The number of iteration at the end of the iteration
    loss_value: float
        Loss value at the end of the iteration
    """

    converged: bool
    failed: bool
    num_total_iterations: int
    loss_value: float


class LBfgsOptimizer:
    """
    Class that performs optimization of a Keras model using L-BFGS in TensorFlow Probability

    Parameters
    ----------
    model: keras.models.Model

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

        # Calculates and stores the indices required to sort the parameters of the Keras model into a one-dimensional array.
        stich_indices, partition_indices = self._generate_weight_indices()
        self._stitch_indices = stich_indices
        self._partition_indices = partition_indices

        # A variable that holds the results of the most recent optimization
        # This is used to restart from the state at the end of the previous execution when executing the optimize method multiple times.
        self._previous_optimizer_results: Optional[
            tfp.optimizer.lbfgs.LBfgsOptimizerResults
        ] = None

    @property
    def weight_tensors(self) -> List:
        """
        Weights for the keras model
        """
        return self._model.trainable_weights

    @property
    def flatten_weight_array(self) -> tf.Tensor:
        """
        A one-dimensional array of weights for the Keras model
        """
        return tf.dynamic_stitch(self._stitch_indices, self.weight_tensors)

    def _generate_weight_indices(self):
        """
        Calculates the index for converting between the weights of a Keras model and a one-dimensional array of those weights. 

        Returns
        -------
        stitch_indices: List[int]
            Index used to sort the weights of the keras model into a one-dimensional array using the tf.dynamic_stitch method.
        partition_indices: List[int]
            Index used to rearrange a one-dimensional array to the same shape as the weight tensor of the Keras model using the tf.dynamic_partition method.

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
        Updates the components of the one-dimensional array passed to the weight of the keras model held by this object.

        Parameters
        ----------
        flatten_new_weights_array : tf.Tensor
            One-dimensional tensor that stores the new value of the weight.
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
        """Execute  L-BFGS optimization
        L-BFGS will terminate when the internal termination condition is met or the specified maximum number of iterations is reached.

        Parameters
        ----------
        max_iterations : int
            Maximum number of iteration

        Returns
        -------
        LBfgsResultState
            Return the state when L-BFGS is terminated
        """
        # Generates a function to pass to tfp.optimizer.lbfgs_optimize
        def value_and_gradients_function(model_weight_array):
            """
            Function to pass to the valu_and_gradients_function argument of tfp.optimizer.lbfgs_optimize

            """
            self._update_model_weights(model_weight_array)

            loss_value, gradients = self._loss_function(**self._loss_function_kwargs)

            flatten_gradients_array = tf.dynamic_stitch(self._stitch_indices, gradients)

            return loss_value, flatten_gradients_array

        if self._previous_optimizer_results is None:
            # The process when you execute the optimize function for the first time
            results = tfp.optimizer.lbfgs_minimize(
                value_and_gradients_function,
                initial_position=self.flatten_weight_array,
                max_iterations=max_iterations,
                **self._lbfgs_kwargs
            )
        else:
            # The process when the optimize function is executed for the second time or later.
            results = tfp.optimizer.lbfgs_minimize(
                value_and_gradients_function,
                initial_position=None,
                max_iterations=max_iterations,
                previous_optimizer_results=self._previous_optimizer_results,
                **self._lbfgs_kwargs
            )

        # The end state is retained. It is used the next optimization is executed.
        self._previous_optimizer_results = results

        # The model weights are updated with the solution at the end of the process.
        self._update_model_weights(results.position)

        # The end state is summarized.
        result_state = LBfgsResultState(
            converged=bool(results.converged),
            failed=bool(results.failed),
            num_total_iterations=int(results.num_iterations),
            loss_value=float(results.objective_value),
        )

        # The end state is retained.
        return result_state
