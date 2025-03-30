"""
This module implements a solver that trains networks
"""

from typing import Any, Dict, List, Tuple, Union

import tensorflow as tf

from .lbfgs import LBfgsOptimizer, LBfgsResultState
from .loss import LossFunction
from .solver_callback import SolverStepEndCallback


class PINNSolver:
    """
    Class to execute the training of the PINN

    Parameters
    ----------
    model: tf.keras.Model
        NN model
    optimizer: tf.keras.optimizers.Optimizer
        Optimized algorithm for network parameters
    loss_functions: Dict[str, LossFunction], key: value = Name: Object for loss function
        Loss function for the network training. The name is used as a label for mapping to loss_weights and for recording history during training.
    loss_weights: Dict[str, float], key: value = Name: Weights
        Weighting of each term in the loss function. Terms not included in this dictionary are added to the overall loss with a weighting of 1.
    metrics: Dict[str, LossFunction], key: value = Name: Object for loss function
        An evaluation function used to monitor the progress of training. It does not affect NN training.
        The name is used as a label when recording history during training. Please be careful not to duplicate the name with loss_functions.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        *,
        optimizer: tf.keras.optimizers.Optimizer,
        loss_functions: Dict[str, LossFunction],
        loss_weights: Dict[str, float] = {},
        metrics: Dict[str, LossFunction],
    ):
        self._model = model
        self._optimizer = optimizer
        self._loss_functions: Dict[str, LossFunction] = loss_functions
        self._loss_weights = loss_weights
        self._metric_functions: Dict[str, LossFunction] = metrics

        self.step_counter: int = 0

        self.histories: Dict[int, Dict[str, float]]
        self.histories = dict()  # {step_number: {metric_name: value}}

        self._TOTAL_TRAIN_LOSS_NAME = "train_loss"

        self._DEFAULT_LOSS_WEIGHT = 1.0
        return

    def solve(self, n_steps: int, callbacks: List[SolverStepEndCallback] = list()):
        """
        Performs network training for the specified number of iterations.

        Parameters
        ----------
        n_steps: int
            The number of iteration
        callbacks: List[SolverStepEndCallback]
            List of callbacks to be called at the end of each iteration
        """
        for _ in range(n_steps):
            # Counter
            self.step_counter += 1

            # Run training
            loss_value_dict = self._train_step()

            # Evaluation
            metric_values_dict = self._eval_step()

            # History tracking
            self.histories[self.step_counter] = {
                key: float(value.numpy()) for key, value in loss_value_dict.items()
            }

            self.histories[self.step_counter].update(
                {key: float(value.numpy()) for key, value in metric_values_dict.items()}
            )

            # Callback
            for callback in callbacks:
                if not callback.should_call(current_step_count=self.step_counter):
                    continue

                callback.on_step_end(
                    model=self._model,
                    loss_and_metrics=self.get_last_loss_and_metrics(),
                    current_step_count=self.step_counter,
                )

    def get_last_loss_and_metrics(self) -> Dict[str, float]:
        """
        Obtain a dictionary that stores the values of the loss function and the evaluation function in the last iteration performed by this solver.
        """
        return self.histories[self.step_counter].copy()

    @tf.function
    def _train_step(self) -> Dict[str, tf.Tensor]:
        """
        A single training step is executed by the function

        Returns
        -------
        loss_values_dict: Dict[str, tf.Tensor]
            A dictionary containing the terms of the loss function and their weighted sums ("train_loss")
        """
        # Initialize the dictionary
        loss_value_dict: Dict[str, Union[float, tf.Tensor]] = {}

        with tf.GradientTape() as tape:
            total_loss_value = 0.0
            loss_value_dict[self._TOTAL_TRAIN_LOSS_NAME] = (
                total_loss_value  
            )

            for loss_name, loss_func in self._loss_functions.items():
                loss_weight = self._loss_weights.get(
                    loss_name, self._DEFAULT_LOSS_WEIGHT
                )
                loss_value = loss_func.calculate_loss(self._model)
                total_loss_value += +loss_weight * loss_value
                loss_value_dict[loss_name] = loss_value

            loss_value_dict[self._TOTAL_TRAIN_LOSS_NAME] = total_loss_value

        # Network parameter update based on gradients of the loss
        loss_gradients = tape.gradient(
            total_loss_value, self._model.trainable_variables
        )
        
        ind = 0
        for var, grad in zip(self._model.trainable_variables, loss_gradients):
            if grad is not None:
                max_grad = tf.reduce_max(tf.abs(grad))
                is_grad_nan = tf.reduce_any(tf.math.is_nan(grad))
                if is_grad_nan:
                    loss_gradients[ind] = tf.zeros_like(grad)
            ind += 1
        
        self._optimizer.apply_gradients(
            zip(loss_gradients, self._model.trainable_variables)
        )

        return loss_value_dict

    @tf.function
    def _eval_step(self) -> Dict[str, tf.Tensor]:
        """
        Execute evaluation

        Returns
        -------
        metric_values_dict: Dict[str, tf.Tensor]
            A dictionary containing the values of each evaluation function
        """
        metric_values_dict: Dict[str, tf.Tensor] = {}

        for metric_name, metric_func in self._metric_functions.items():
            metric_value = metric_func.calculate_loss(self._model)
            metric_values_dict[metric_name] = metric_value

        return metric_values_dict


class PINNSolverLBfgs:
    """
    L-BFGS Solver

    Parameters
    ----------
    model: tf.keras.Model
        Network to be trained
    loss_functions: Dict[str, LossFunction], key: value = Name: Loss object
        The loss function used to train the NN. The name is used as a mapping to loss_weights and as a label for recording the history during training.
    loss_weights: Dict[str, float], key: value = Name: Weighting value
        Weighting of each term in the loss function. Terms not included in this dictionary are added to the overall loss with a weighting of 1.
    metrics: Dict[str, LossFunction], key: value = Name: Loss object
        An evaluation function used to observe the progress of training, which does not affect the training of NN.
       The name is used as a label for recording the history during training; be careful not to duplicate the name with loss_functions.
    lbfgs_kwargs: Dict[str, Any]
        L-BFGS option, passed as an argument to tfp.optimize.lbfgs_minimize().
    """

    def __init__(
        self,
        model: tf.keras.Model,
        *,
        loss_functions: Dict[str, LossFunction],
        loss_weights: Dict[str, float] = {},
        metrics: Dict[str, LossFunction],
        lbfgs_kwargs: Dict[str, Any] = {},
    ):

        self._model = model
        self._loss_functions: Dict[str, LossFunction] = loss_functions
        self._loss_weights = loss_weights
        self._metric_functions: Dict[str, LossFunction] = metrics

        self.step_counter: int = 0

        self.histories: Dict[int, Dict[str, float]]
        self.histories = dict()  # {step_number: {metric_name: value}}

        self._TOTAL_TRAIN_LOSS_NAME = "train_loss"

        self._DEFAULT_LOSS_WEIGHT = 1.0

        def loss_gradient_function():
            with tf.GradientTape() as tape:
                total_loss_value = 0.0

                for loss_name, loss_func in self._loss_functions.items():
                    loss_weight = self._loss_weights.get(
                        loss_name, self._DEFAULT_LOSS_WEIGHT
                    )
                    loss_value = loss_func.calculate_loss(self._model)
                    total_loss_value += +loss_weight * loss_value

            loss_gradients = tape.gradient(
                total_loss_value, self._model.trainable_variables
            )

            return total_loss_value, loss_gradients

        self._optimizer = LBfgsOptimizer(
            model=self._model,
            loss_function=loss_gradient_function,
            loss_function_kwargs=dict(),
            lbfgs_kwargs=lbfgs_kwargs,
        )

        return

    def solve(self, n_steps: int, callbacks: List[SolverStepEndCallback] = list()):
        """
        Perform network training until the specified number of iterations is reached.

        Parameters
        ----------
        n_steps: int
            Number of iteration
        callbacks: List[SolverStepEndCallback]
            List of callbacks to be called at the end of each iteration
        """
        for _ in range(n_steps):
            # Add 1 on the counter
            self.step_counter += 1

            # Perform one iteration of the training.
            lbfgs_result_state, loss_value_dict = self._train_step()

            # Evaluation
            metric_values_dict = self._eval_step()

            # History tracking
            self.histories[self.step_counter] = {
                key: float(value.numpy()) for key, value in loss_value_dict.items()
            }

            self.histories[self.step_counter].update(
                {key: float(value.numpy()) for key, value in metric_values_dict.items()}
            )

            if lbfgs_result_state.converged:
                # L-BFGS training was converged
                print(f"[L-BFGS] converged: state={lbfgs_result_state}.")

                for callback in callbacks:
                    callback.on_step_end(
                        model=self._model,
                        loss_and_metrics=self.get_last_loss_and_metrics(),
                        current_step_count=self.step_counter,
                    )
                break

            elif lbfgs_result_state.failed:
                # If a failure occurs in L-BFGS, the training ends here.
                # The L-BFGS for TensorFlow Probability fails in the following conditions:
                # "a line search step failed to find a suitable step size satisfying Wolfe conditions"
                # (https://www.tensorflow.org/probability/api_docs/python/tfp/optimizer/lbfgs_minimize)
                print(f"[L-BFGS] line search step failed: state={lbfgs_result_state}.")

                for callback in callbacks:
                    callback.on_step_end(
                        model=self._model,
                        loss_and_metrics=self.get_last_loss_and_metrics(),
                        current_step_count=self.step_counter,
                    )
                break

            else:
                # If L-BFGS has neither converged nor failed, continue training.

                for callback in callbacks:
                    if not callback.should_call(current_step_count=self.step_counter):
                        continue

                    # Callback
                    callback.on_step_end(
                        model=self._model,
                        loss_and_metrics=self.get_last_loss_and_metrics(),
                        current_step_count=self.step_counter,
                    )

    def get_last_loss_and_metrics(self) -> Dict[str, float]:
        """
        Obtains a dictionary containing the values of the loss function and the evaluation function at the last iteration performed by this solver.
        """
        return self.histories[self.step_counter].copy()

    def _train_step(self) -> Tuple[LBfgsResultState, Dict[str, tf.Tensor]]:
        """
        Perform 1 step of training.

        Returns
        -------
        result_state: LBfgsResultState:
            State at end of L-BFGS iteration
        loss_values_dict: Dict[str, tf.Tensor]
            A dictionary containing the terms of the loss function and their weighted sums ("train_loss")
        """
        result_state = self._optimizer.optimize(max_iterations=self.step_counter)

        loss_value_dict = self._calculate_losses()

        return result_state, loss_value_dict

    @tf.function
    def _calculate_losses(self) -> Dict[str, tf.Tensor]:
        # Initializes a dictionary to store loss function values.
        loss_value_dict: Dict[str, Union[float, tf.Tensor]] = {}

        total_loss_value = 0.0
        loss_value_dict[self._TOTAL_TRAIN_LOSS_NAME] = (
            total_loss_value
        )

        for loss_name, loss_func in self._loss_functions.items():
            loss_weight = self._loss_weights.get(loss_name, self._DEFAULT_LOSS_WEIGHT)
            loss_value = loss_func.calculate_loss(self._model)
            total_loss_value += +loss_weight * loss_value
            loss_value_dict[loss_name] = loss_value

        loss_value_dict[self._TOTAL_TRAIN_LOSS_NAME] = total_loss_value

        return loss_value_dict

    @tf.function
    def _eval_step(self) -> Dict[str, tf.Tensor]:
        """
        Perform evaluation

        Returns
        -------
        metric_values_dict: Dict[str, tf.Tensor]
            A dictionary containing the values of each evaluation function
        """
        metric_values_dict: Dict[str, tf.Tensor] = {}

        for metric_name, metric_func in self._metric_functions.items():
            metric_value = metric_func.calculate_loss(self._model)
            metric_values_dict[metric_name] = metric_value

        # Returns a dictionary that records the value of the evaluation function
        return metric_values_dict
