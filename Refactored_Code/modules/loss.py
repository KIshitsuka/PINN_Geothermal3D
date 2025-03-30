"""
Module for loss function
"""

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from modules.normalizer import Normalizer
from .pred_physical_property import Densw_pred, HCw_pred, Viscow_pred
from modules import rearrange_coordinate_container as rcc


def swap_with_axis0(tensor: tf.Tensor, axis: int) -> tf.Tensor:

    rank = len(tensor.shape)

    perm = list(range(rank))

    perm[0], perm[axis] = axis, 0

    return tf.transpose(tensor, perm=perm)

def derivative_1d_FD(in_tensor:tf.Tensor,
                    grid_shape: tuple,
                    dx: float, axis: int) -> tf.Tensor:

    # Convert 1D array into 3D grid
    F = rcc.rearrange_list_to_grid(arr_list=in_tensor, grid_shape=grid_shape)

    F = swap_with_axis0(F, axis)

    # 4th-order central differentiation and 3rd-order forward/backward differentiation
    dFdx_in = (- F[4:, :, :, :] + 8 * F[3:-1, :, :, :]
                - 8 * F[1:-3, :, :, :] + F[0:-4,:, :, :]) / 12 / dx
    dFdx_left=(- 11 * F[0:-3, :, :, :] + 18 * F[1:-2, :, :, :]
                - 9 * F[2:-1, :, :, :] + 2 * F[3:, :, :, :]) / 6 / dx
    dFdx_left = dFdx_left[0:2, :, :, :]
    dFdx_right=(11 * F[3:, :, :, :] - 18 * F[2:-1, :, :, :]
                + 9 * F[1:-2, :, :, :] - 2 * F[0:-3, :, :, :]) / 6 / dx
    dFdx_right = dFdx_right[-2:, :, :, :]
    dFdx = tf.concat([dFdx_left, dFdx_in, dFdx_right], axis=0)

    dFdx = swap_with_axis0(dFdx, axis)

    dFdx_list = rcc.rearrange_grid_to_list(dFdx)

    return dFdx_list


class LossFunction(ABC):
    """
    Class to represent loss function

    """

    @abstractmethod
    def calculate_loss(self, model: tf.keras.Model) -> tf.Tensor:
        pass


class ObservationLoss(LossFunction):
    """
    Loss function for the misfit between observations and predictions

    Parameters
    ----------
    observed_point_coordinates: np.ndarray, shape=(Num_points, Dim_coordinates)
        Corrdinates of observations
    observed_values: np.ndarray
        True value at each observation point
    NN_output_index: int
        Indices of NN outputs
    dtype:
        Data type
    """

    def __init__(
        self,
        observed_point_coordinates: np.ndarray,
        observed_values: np.ndarray,
        NN_output_index: int,
        dtype=tf.float32,
    ):

        self.observed_point_coordinates = tf.constant(
            observed_point_coordinates, dtype=dtype
        )
        self.observed_values = tf.constant(observed_values, dtype=dtype)
        self.NN_output_index = NN_output_index
        return

    def calculate_loss(self, model: tf.keras.Model) -> tf.Tensor:
        pred_values = model(self.observed_point_coordinates)[self.NN_output_index]

        loss_value = tf.reduce_mean(tf.square(self.observed_values - pred_values))
        return loss_value


class NeumannBoundaryConditionLoss(LossFunction):
    """
    Loss function for the Neumann boundary condition

    Parameters
    ----------
    boundary_coordinates: np.ndarray, shape=(Num_points, Dim_coordinates)
        Coordinates of the points at the boundaries
    boundary_values: np.ndarray
        True values at the boundaries
    normal_vector: np.ndarray
        Normal vector of each point at the boundaries
    normalizer: Normalizer
        Normalizer of the quantities used in the network
    NN_output_index: int
        Indices of NN outputs
    dtype:
        Data type
    """

    def __init__(
        self,
        boundary_coordinates: np.ndarray,
        boundary_values: np.ndarray,
        normal_vector: np.ndarray,
        normalizer: Normalizer,
        NN_output_index: int,
        dtype=tf.float32,
    ):

        self.boundary_coordinates = tf.constant(boundary_coordinates, dtype=dtype)
        self.boundary_values = tf.constant(boundary_values, dtype=dtype)
        self.normal_vector = tf.constant(normal_vector, dtype=dtype)
        self.normalizer = normalizer
        self.NN_output_index = NN_output_index

    def calculate_loss(self, model: tf.keras.Model):
        x1 = self.boundary_coordinates[:, 0]
        x2 = self.boundary_coordinates[:, 1]
        x3 = self.boundary_coordinates[:, 2]
        normal_1 = self.normal_vector[:, 0:1]
        normal_2 = self.normal_vector[:, 1:2]
        normal_3 = self.normal_vector[:, 2:3]
        with tf.GradientTape() as tape11, tf.GradientTape() as tape12, tf.GradientTape() as tape13:
            tape11.watch(x1)
            tape12.watch(x2)
            tape13.watch(x3)
            pred = model(tf.stack([x1, x2, x3], axis=1))[self.NN_output_index]
            denormalized_pred = self.normalizer.denormalize(pred)
        g_1 = tape11.gradient(denormalized_pred, x1)
        g_2 = tape12.gradient(denormalized_pred, x2)
        g_3 = tape13.gradient(denormalized_pred, x3)
        g = (g_1 * normal_1 + g_2 * normal_2 + g_3 * normal_3)

        loss_value = tf.reduce_mean(tf.square(tf.subtract(self.boundary_values, g)))
        return loss_value


class DirichletBoundaryConditionLoss(LossFunction):
    """
    Loss function for the Dirichlet boundary condition

    Parameters
    ----------
    boundary_coordinates: np.ndarray, shape=(Num_points, Dim_coordinates)
        Coordinates of the points at the boundaries
    boundary_values: np.ndarray
        True values at the boundaries
    NN_output_index: int
        Indices of NN outputs
    dtype:
        Data type
    """

    def __init__(
        self,
        boundary_coordinates: np.ndarray,
        boundary_values: np.ndarray,
        NN_output_index: int,
        dtype=tf.float32,
    ):

        self.boundary_coordinates = tf.constant(boundary_coordinates, dtype=dtype)
        self.boundary_values = tf.constant(boundary_values)
        self.NN_output_index = NN_output_index

    def calculate_loss(self, model: tf.keras.Model):
        pred = model(self.boundary_coordinates)[self.NN_output_index]

        loss_value = tf.reduce_mean(tf.square(tf.subtract(self.boundary_values, pred)))
        return loss_value


class PhysicsInformedLoss_r1(LossFunction):
    """
    Loss function for the mass conservation

    Parameters
    ----------
    collocation_point_coordinates: np.ndarray, shape=(Num_points, Dim_coordinates)
        Cooorindates of target points
    T_normalizer, p_normalizer, k_normalizer: Normalizer
        Normalizers for temperature, pressure, logarithm of permeability
    dtype:
        Data type
    """

    def __init__(
        self,
        collocation_point_coordinates: np.ndarray,
        T_normalizer: Normalizer,
        p_normalizer: Normalizer,
        k_normalizer: Normalizer,
        dtype=tf.float32,
    ):

        self.collocation_point_coordinates = tf.constant(
            collocation_point_coordinates, dtype=dtype
        )
        self.T_normalizer = T_normalizer
        self.p_normalizer = p_normalizer
        self.k_normalizer = k_normalizer

    def calculate_loss(self, model: tf.keras.Model):
        x1 = self.collocation_point_coordinates[:, 0]
        x2 = self.collocation_point_coordinates[:, 1]
        x3 = self.collocation_point_coordinates[:, 2]

        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x1)
            tape1.watch(x2)
            tape1.watch(x3)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(x1)
                tape2.watch(x2)
                tape2.watch(x3)

                T, p, k = model(tf.stack([x1, x2, x3], axis=1))
                pdenm = self.p_normalizer.denormalize(p)

            p_x1 = tape2.gradient(pdenm, x1)
            p_x2 = tape2.gradient(pdenm, x2)
            p_x3 = tape2.gradient(pdenm, x3)
            Tdenm = self.T_normalizer.denormalize(T)
            kdenm = self.k_normalizer.denormalize(k)
            dens = Densw_pred(Tdenm, pdenm)
            visc = Viscow_pred(Tdenm, dens)
            densg = dens * 9.8
            ddv = tf.math.divide(dens, visc)
            pf_x1 = tf.math.multiply(ddv, p_x1)
            pf_x2 = tf.math.multiply(ddv, p_x2)
            pf_x3tp = tf.math.subtract(p_x3, densg)
            pf_x3 = tf.math.multiply(ddv, pf_x3tp)
            g1 = tf.math.multiply(10**kdenm, pf_x1)
            g2 = tf.math.multiply(10**kdenm, pf_x2)
            g3 = tf.math.multiply(10**kdenm, pf_x3)

        f1tp = tape1.gradient(g1, x1)
        f2tp = tape1.gradient(g2, x2)
        f3tp = tape1.gradient(g3, x3)

        posf1 = tf.math.is_finite(f1tp)
        posf2 = tf.math.is_finite(f2tp)
        posf3 = tf.math.is_finite(f3tp)
        f1 = tf.where(posf1, f1tp, [10**5])
        f2 = tf.where(posf2, f2tp, [10**5])
        f3 = tf.where(posf3, f3tp, [10**5])

        del tape2
        del tape1

        r1_value = self.fun_r1(f1, f2, f3)

        loss_value = tf.reduce_mean(tf.square(r1_value))
        return loss_value

    def fun_r1(self, f1, f2, f3):
        """Residual of the PDE"""
        return f1 + f2 + f3


class PhysicsInformedLoss_r2(LossFunction):
    """
    Loss function for the energy conservation

    Parameters
    ----------
    collocation_point_coordinates: np.ndarray, shape=(Num_points, Dim_coordinates)
        A list of corrdinates
    T_normalizer, p_normalizer, k_normalizer: Normalizer
        Normalizer for each physical quantity
    Lambda
        Thermal conductivity
    dtype:
        Data type
    """

    def __init__(
        self,
        collocation_point_coordinates: np.ndarray,
        T_normalizer: Normalizer,
        p_normalizer: Normalizer,
        k_normalizer: Normalizer,
        Lambda: float,
        dtype=tf.float32,
    ):

        self.collocation_point_coordinates = tf.constant(
            collocation_point_coordinates, dtype=dtype
        )
        self.T_normalizer = T_normalizer
        self.p_normalizer = p_normalizer
        self.k_normalizer = k_normalizer
        self.Lambda = Lambda

    def calculate_loss(self, model: tf.keras.Model):
        x1 = self.collocation_point_coordinates[:, 0]
        x2 = self.collocation_point_coordinates[:, 1]
        x3 = self.collocation_point_coordinates[:, 2]
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x1)
            tape1.watch(x2)
            tape1.watch(x3)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(x1)
                tape2.watch(x2)
                tape2.watch(x3)

                T, p, k = model(tf.stack([x1, x2, x3], axis=1))
                Tdenm = self.T_normalizer.denormalize(T)
                pdenm = self.p_normalizer.denormalize(p)

            p_x1 = tape2.gradient(pdenm, x1)
            p_x2 = tape2.gradient(pdenm, x2)
            p_x3 = tape2.gradient(pdenm, x3)
            T_x1 = tape2.gradient(Tdenm, x1)
            T_x2 = tape2.gradient(Tdenm, x2)
            T_x3 = tape2.gradient(Tdenm, x3)
            kdenm = self.k_normalizer.denormalize(k)
            dens = Densw_pred(Tdenm, pdenm)
            visc = Viscow_pred(Tdenm, dens)

            hcw = HCw_pred(Tdenm, pdenm)
            densg = dens * 9.8
            ddw_tp1 = tf.math.divide(dens, visc)
            ddw_tp2 = tf.math.multiply(hcw, ddw_tp1)
            ddw = tf.math.multiply(Tdenm, ddw_tp2)
            pf_x1 = tf.math.multiply(ddw, p_x1)
            pf_x2 = tf.math.multiply(ddw, p_x2)
            pf_x3tp = tf.math.subtract(p_x3, densg)
            pf_x3 = tf.math.multiply(ddw, pf_x3tp)
            g1 = tf.math.multiply(10**kdenm, pf_x1)
            g2 = tf.math.multiply(10**kdenm, pf_x2)
            g3 = tf.math.multiply(10**kdenm, pf_x3)

        f_x1x1tp = tape1.gradient(g1, x1)
        f_x2x2tp = tape1.gradient(g2, x2)
        f_x3x3tp = tape1.gradient(g3, x3)

        posf1 = tf.math.is_finite(f_x1x1tp)
        posf2 = tf.math.is_finite(f_x2x2tp)
        posf3 = tf.math.is_finite(f_x3x3tp)
        f_x1x1 = tf.where(posf1, f_x1x1tp, [10**5])
        f_x2x2 = tf.where(posf2, f_x2x2tp, [10**5])
        f_x3x3 = tf.where(posf3, f_x3x3tp, [10**5])

        T_x1x1 = tape1.gradient(T_x1, x1)
        T_x2x2 = tape1.gradient(T_x2, x2)
        T_x3x3 = tape1.gradient(T_x3, x3)

        del tape2
        del tape1

        r2_value = self.fun_r2(f_x1x1, f_x2x2, f_x3x3, T_x1x1, T_x2x2, T_x3x3)
        loss_value = tf.reduce_mean(tf.square(r2_value))
        return loss_value

    def fun_r2(self, f_x1x1, f_x2x2, f_x3x3, T_x1x1, T_x2x2, T_x3x3):
        """Residual of the PDE"""
        return f_x1x1 + f_x2x2 + f_x3x3 + self.Lambda * (T_x1x1 + T_x2x2+ T_x3x3)
