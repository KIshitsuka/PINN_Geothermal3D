import numpy as np


def get_wells_neighbor_indecies(X_wells_in, X_p, depth, N_neighbor):
    """
    This function returns the indices of the grid points 
    in the vicinity of the well coordinates for training and validation, respectively.

    Parameters
    ----------
        X_wells_in: nd.array
                  Well coordinates
                  shape = (Num_well_points, 3)
        X_p: nd.array
                  The coordinates used for interpolation.
                  shape = (Num_points_interpolation, 3)
        depth: float
              The threshold of z axis between training and validation data
        N_neighbor: int
              Num_neighboring_points around wells used for interpolation
    Outputs
    --------
        idx_tpk_train: nd.array(dtype = int)
                      List of indices for the neighboring points used for training
                      shape = (Num_neighboring_points)
        idx_tpk_val: nd.array(dtype = int)
                      List of indices for the neighboring points used for validation
                      shape = (Num_neighboring_points)
    """
    N_pt_wells = X_wells_in.shape[0]
    N_train = 0
    N_val = 0
    idx_tpk_train = []
    idx_tpk_val = []
    X_wells_train = []
    X_wells_val = []
    nearest_idx_to_train = []
    nearest_idx_to_val = []
    for idx in range(N_pt_wells):
      distances = np.linalg.norm(X_p - X_wells_in[idx, :], axis = 1)
      idx_neighbor_list = np.argsort(distances)[:N_neighbor]
      if X_wells_in[idx, 2] <= depth: #validation data
        nearest_idx_to_val.append(idx_neighbor_list)
        X_wells_val.append(X_wells_in[idx, :])
        for idx_neighbor in idx_neighbor_list:
          if idx_neighbor not in idx_tpk_val:
            idx_tpk_val.append(idx_neighbor)
            N_val = N_val + 1
      else:
        nearest_idx_to_train.append(idx_neighbor_list)
        X_wells_train.append(X_wells_in[idx, :])
        for idx_neighbor in idx_neighbor_list:
          if idx_neighbor not in idx_tpk_train:
            idx_tpk_train.append(idx_neighbor)
            N_train = N_train + 1
    return np.array(idx_tpk_train, dtype = int), np.array(idx_tpk_val, dtype = int)


def get_training_and_validation_data(X_wells_in, T_wells_in, p_wells_in, k_wells_in, depth):
    """
    Function that separates the coordinates and physical quantities of the well into training and validation sets
    Parameters
    -----------
        X_wells_in: nd.array
                    Coordinates_well
                    shape = (Num_points_well, 3)
        T_wells_in: nd.array
                    Temperatures at wells
                    shape = (Num_points_well, 1)
        p_wells_in: nd.array
                    Pressures at wells
                    shape = (Num_points_well, 1)
        k_wells_in: nd.array
                    Permeabilities at wells
                    shape = (Num_points_well, 1)
        depth: float
              Threshold of Z coordinates between training and validation data

    Outputs
    --------
        *_wells_train,m *_wells_val : nd.array
                  Returns the coordinates and physical quantities of the wells, which are divided into training and validation.

    """
    N_pt_wells = X_wells_in.shape[0]

    X_wells_train = []
    X_wells_val = []

    T_wells_train = []
    T_wells_val = []

    p_wells_train = []
    p_wells_val = []

    k_wells_train = []
    k_wells_val = []
    for idx in range(N_pt_wells):
      if X_wells_in[idx, 2] <= depth: #validation data
        X_wells_val.append(X_wells_in[idx, :])
        T_wells_val.append(T_wells_in[idx, :])
        p_wells_val.append(p_wells_in[idx, :])
        k_wells_val.append(k_wells_in[idx, :])
      else: # training data
        X_wells_train.append(X_wells_in[idx, :])
        T_wells_train.append(T_wells_in[idx, :])
        p_wells_train.append(p_wells_in[idx, :])
        k_wells_train.append(k_wells_in[idx, :])

    return np.array(X_wells_train), np.array(T_wells_train), \
           np.array(p_wells_train), np.array(k_wells_train),\
           np.array(X_wells_val), np.array(T_wells_val), \
           np.array(p_wells_val), np.array(k_wells_val)
