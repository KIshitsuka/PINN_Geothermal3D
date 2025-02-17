from scipy.interpolate import Rbf

def get_Tkp_on_new_coords(X_old, T_old, k_old, p_old, X_new):
    """
    This function takes physical quantities (T_old, k_old, p_old) 
    in the old coordinates (X_old) and 
    uses the Rbf interpolation formula to calculate and 
    return physical quantities (T_new, k_new, p_new) in the new coordinates (X_new).
    
    Parameters
    ----------
        X_old: nd.array
               Old coordinates
               shape = (Num_old_coordinates, 3)
        T_old: nd.array
               Temperatures at the old coordinates
               shape = (Num_old_coordinates, 1)
        k_old: nd.array
               Permeabilities at the old coordinates
               shape = (Num_old_coordinates, 1)
        p_old: nd.array
               Pressures at the old coordinates
               shape = (Num_old_coordinates, 1)
        X_new: nd.array
               New coordinates to evaluate physical qunatities
               shape = (Num_new_coordinates, 3)
    Outputs
    --------
        T_new : nd.array
                Temperatures at the new coordinates
                shape = (Num_new_coordinates)
        k_new : nd.array
                Permeabilities at the new coordinates
                shape = (Num_new_coordinates)
        p_new : nd.array
                Pressures at the new coordinates
                shape = (Num_new_coordinates)

    """    

    # scipy.interpolate.Rbf is used to create an interpolation formula.
    rbfi = Rbf(X_old[:, 0], X_old[:,1], X_old[:, 2], T_old[:, 0])

    # Obtain the physical quantity on the new coordinates.
    T_new = rbfi(X_new[:,0], X_new[:,1], X_new[:, 2])

    rbfi = Rbf(X_old[:, 0], X_old[:,1], X_old[:,2], k_old[:, 0])
    k_new = rbfi(X_new[:,0], X_new[:,1], X_new[:, 2])

    rbfi = Rbf(X_old[:, 0], X_old[:,1], X_old[:, 2], p_old[:, 0])
    p_new = rbfi(X_new[:,0], X_new[:,1], X_new[:, 2])

    return T_new, k_new, p_new

