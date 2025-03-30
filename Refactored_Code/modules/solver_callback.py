"""
This module implements a callback that is called at the end of each iteration during training.
"""

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from modules.normalizer import Normalizer


class SolverStepEndCallback(ABC):
    """
    Base class for callbacks that are called at the end of each iteration of the PINN solver.

    """

    def should_call(self, current_step_count: int) -> bool:
        """
        Takes the number of iterations and returns whether this callback should be executed.

        Parameters
        ----------
        current_step_count: int
            Current number of iteration

        Returns
        -------
        bool
            True if this is the number of iterations the callback should be called, False otherwise
        """
        return True

    @abstractmethod
    def on_step_end(
        self,
        model: tf.keras.Model,
        loss_and_metrics: Dict[str, float],
        current_step_count: int,
    ):
        """
        A callback that is called at the end of each iteration during the solver's execution in which should_call is True.

        Parameters
        ----------
        model: tf.keras.Model
            Current network model
        loss_and_metrics: Dict[str, float]
            Values of loss function and evaluation function in this iteration
        current_step_count: int
            Current number of iteration
        """
        pass


class PrintCurrentLossAndMetricsCallback(SolverStepEndCallback):
    """
    A callback that prints the values of the loss function and evaluation function at each point in time at regular intervals during training.

    Parameters
    ----------
    interval: int
        Interval between calls of this callback
    """

    def __init__(self, interval: int = 1):
        self._interval = interval

    def should_call(self, current_step_count: int) -> bool:
        return current_step_count % self._interval == 0

    def on_step_end(self, model, loss_and_metrics, current_step_count):
        loss_and_metrics_text = ", ".join(
            [f"{name}={value:.2e}" for name, value in loss_and_metrics.items()]
        )
        print(f"[{current_step_count}] {loss_and_metrics_text}")


class CsvLoggerCallback(SolverStepEndCallback):
    """
    A callback that records the loss function and evaluation function for each iteration in a CSV during training.

    Parameters
    ----------
    csv_path: str or Path
        Save path of thw CSV file
    """

    def __init__(self, csv_path: Union[str, Path]):
        self.csv_path = csv_path

        self.is_first_call = True

    def on_step_end(self, model, loss_and_metrics, current_step_count):
        if self.is_first_call:
            self.loss_and_metrics_name_list = list(loss_and_metrics.keys())

            with open(self.csv_path, mode="w", encoding="utf_8_sig") as csvfile:
                writer = csv.writer(
                    csvfile, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n"
                )
                writer.writerow(["step"] + self.loss_and_metrics_name_list)

            self.is_first_call = False

        value_list = [current_step_count]
        value_list += [
            loss_and_metrics[name] for name in self.loss_and_metrics_name_list
        ]

        with open(self.csv_path, mode="a", encoding="utf_8_sig") as csvfile:
            writer = csv.writer(
                csvfile, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n"
            )
            writer.writerow(value_list)


class NNWeightsCheckpointCallback(SolverStepEndCallback):
    """
    Callback to save NN weights during training.
    TensorFlow's Checkpoint format is used to save the weights.

    Parameters
    ----------
    checkpoint_dir: str or Path
        Path to save checkpoints
    interval: int
        Interval to call this callback
    checkpoint_name: str
        Name of the checkpoint
    max_to_keep: int
        Maximum number of checkpoints to hold
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        interval: int,
        checkpoint_name: str = "weights.ckpt",
        max_to_keep: int = 500,
    ):
        self._checkpoint_dir = checkpoint_dir
        self._interval = interval
        self._checkpoint_name = checkpoint_name
        self._max_to_keep = max_to_keep

    def should_call(self, current_step_count: int) -> bool:
        return current_step_count % self._interval == 0

    def on_step_end(self, model, loss_and_metrics, current_step_count):
        # Save checkpoints
        checkpoint = tf.train.Checkpoint(model=model.variables)
        manager = tf.train.CheckpointManager(
            checkpoint,
            directory=self._checkpoint_dir,
            checkpoint_name=self._checkpoint_name,
            max_to_keep=self._max_to_keep,
        )

        path = manager.save(checkpoint_number=current_step_count)
        print(f"weights saved to {path}")


class AllGridPredictionCallback(SolverStepEndCallback):
    """
    Callback to save the NN output values for the entire grid to a CSV file during training.

    Parameters
    ----------
    save_prediction_dir: str or Path
        Save directory
    interval: int
        Interval between executions of this callback
    all_grid_point_coordinates: np.ndarray
        List of coordinates of points on the entire grid
    T_normalizer: Normalizer,
        T normalizer
    p_normalizer: Normalizer,
        p normalizer
    k_normalizer: Normalizer,
        k normalizer
    """

    def __init__(
        self,
        save_prediction_dir: Union[str, Path],
        interval: int,
        all_grid_point_coordinates: np.ndarray,
        T_normalizer: Normalizer,
        p_normalizer: Normalizer,
        k_normalizer: Normalizer,
    ):
        self._save_prediction_dir = Path(save_prediction_dir)
        self._interval = interval
        self._all_grid_point_coordinates = tf.constant(all_grid_point_coordinates)
        self._T_denormalizer = T_normalizer
        self._p_denormalizer = p_normalizer
        self._k_denormalizer = k_normalizer

    def should_call(self, current_step_count: int) -> bool:
        return current_step_count % self._interval == 0

    def on_step_end(self, model, loss_and_metrics, current_step_count):
        T_pred, p_pred, k_pred = model(self._all_grid_point_coordinates)

        denorm_T_pred = self._T_denormalizer.denormalize(T_pred)
        denorm_p_pred = self._p_denormalizer.denormalize(p_pred)
        denorm_k_pred = self._k_denormalizer.denormalize(k_pred)

        self._save_prediction_dir.mkdir(parents=True, exist_ok=True)
        prediction_df = pd.DataFrame(columns=["x1", "x2", "x3", "T_pred", "p_pred", "k_pred"])
        prediction_df[["x1", "x2", "x3"]] = self._all_grid_point_coordinates.numpy()
        prediction_df["T_pred"] = denorm_T_pred.numpy()
        prediction_df["p_pred"] = denorm_p_pred.numpy()
        prediction_df["k_pred"] = denorm_k_pred.numpy()
        prediction_df.to_csv(
            self._save_prediction_dir / f"predicted_{current_step_count}.csv",
            index=False,
        )
