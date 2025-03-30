"""
This module implements the process of reordering the data structure in which coordinates are stored.
"""
import tensorflow as tf


def rearrange_coordinate_list_to_grid(
      coordinate_list: tf.Tensor, n_x1: int, n_x2: int, n_x3: int) -> tf.Tensor:
    """
    Rearranges coordinates stored in a single-column list into a two-dimensional grid.
    The order of the elements follows the Fortran method.

    Parameters
    ----------
    coordinate_list: tf.Tensor
      shape = (num_points, ndim_coordinates)
    n_x1: int
      The number of data points along x1 axis
    n_x2: int
      The number of data points along x2 axis
    n_x3: int
      The number of data points along x3 axis

    Returns
    -------
    coordinate_grid: tf.Tensor
      shape = (n_x1, n_x2, n_x3, ndim_coordinates)
    """
    ndim_coordinates = coordinate_list.shape[-1]
    coordinate_grid = tf.transpose(
        tf.reshape(coordinate_list, [n_x3, n_x2, n_x1, ndim_coordinates]), perm=[1, 0, 2])

    assert coordinate_grid.shape == (n_x1, n_x2, n_x3, ndim_coordinates)
    return coordinate_grid


def rearrange_coordinate_grid_to_list(coordinate_grid: tf.Tensor) -> tf.Tensor:
    """
    Re-arranges the coordinates stored in a 3-dimensional grid into a single-column list.
    The order of the elements follows the fortran method

    Parameters
    ----------
    coordinate_grid: tf.Tensor
      shape = (n_x1, n_x2, n_x3, ndim_coordinates)

    Returns
    -------
    coordinate_list: tf.Tensor
      shape = (n_x1*n_x2*n_x3, ndim_coordinates)
    """
    n_x1, n_x2, n_x3, ndim_coordinates = coordinate_grid.shape
    coordinate_list = tf.reshape(
        tf.transpose(coordinate_grid, perm=[1, 0, 2]), [n_x1*n_x2*n_x3, ndim_coordinates])

    assert coordinate_list.shape == (n_x1*n_x2*n_x3, ndim_coordinates)
    return coordinate_list

def rearrange_grid_to_list(arr_grid: tf.Tensor) -> tf.Tensor:
    """
    Array data stored in a three-dimensional grid is rearranged into a single-column list.
    The ordering of elements follows the fortran method.

    Parameters
    ----------
    arr_grid: tf.Tensor
      shape = (n_x1, n_x2, n_x3, ndim_coordinates)

    Returns
    -------
    arr_list: tf.Tensor
      shape = (n_x1*n_x2*n_x3, ndim_coordinates)
    """
    ndim_coordinates = arr_grid.shape[-1]
    arr_transposed = tf.transpose(arr_grid, perm=[2, 1, 0, 3])
    arr_list = tf.reshape(arr_transposed, shape=(-1, ndim_coordinates))

    return arr_list

def rearrange_list_to_grid(arr_list: tf.Tensor, grid_shape: tuple) -> tf.Tensor:
    """
    Array data stored in a single-column list is rearranged into a three-dimensional grid.
    The ordering of elements follows the fortran method.
    Parameters
    ----------
    arr_list: tf.Tensor
      shape = (num_points, 1)
    grid_shape: tuple
      shape = (n_x1, n_x2, n_x3, ndim_coordinates)

    Returns
    -------
    arr_grid: tf.Tensor
      shape = (n_x1, n_x2, n_x3, ndim_coordinates)
    """
    shape_list = list(grid_shape)
    shape_list[0], shape_list[2] = shape_list[2], shape_list[0]
    arr_transposed_grid = tf.reshape(arr_list, shape=shape_list)
    arr_grid = tf.transpose(arr_transposed_grid, perm=[2, 1, 0, 3])

    return arr_grid
