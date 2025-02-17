def get_boundary_indexlist(N_x1, N_x2, N_x3, X):
    """
    Fuction to obtain boundary indices
    Parameters
    -----------
        N_x1: int
            Number of datapoints in x1 (x) direction
        N_x2: int
            Number of datapoints in x2 (y) direction
        N_x3: int
            Number of datapoints in x3 (z) direction
        X: nd.array
            Coordinates for the target domain
            shape = (N_x1 * N_x2 * N_x3, 3)

    Outputs
    --------
        idx_* : nd.array(dtype = int)
                Return indices for boundaries
                shape = (Num_points_eachbound)
    """


    xmin, ymin, zmin = X.min(0)
    xmax, ymax, zmax = X.max(0)


    idx_up = []
    idx_low = []

    for i in range(N_x1*N_x2*N_x3):
        if (X[i, 2] == zmax): # up
            idx_up.append(i)
        if (X[i, 2] == zmin): #low
            idx_low.append(i)

    idx_west = []
    idx_east = []

    for i in range(N_x1*N_x2*N_x3):
        if (X[i, 0] == xmin): # west
            idx_west.append(i)
        if (X[i, 0] == xmax): # east
            idx_east.append(i)


    idx_south = []
    idx_north = []

    for i in range(N_x1*N_x2*N_x3):
        if (X[i, 1] == ymin): # south
            idx_south.append(i)
        if (X[i, 1] == ymax): # north
            idx_north.append(i)


    return idx_low, idx_up, idx_west, idx_east, idx_south, idx_north
