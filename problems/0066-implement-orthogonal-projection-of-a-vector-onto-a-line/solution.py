
def orthogonal_projection(v, L):
    """
    Compute the orthogonal projection of vector v onto line L.

    :param v: The vector to be projected
    :param L: The line vector defining the direction of projection
    :return: List representing the projection of v onto L
    """
    import numpy as np
    v = np.array(v)
    L = np.array(L)

    num = np.dot(v, L)
    den = np.dot(L, L)
    res = num / den *L

    #res_1 = res * L

    return np.round(res, 3).tolist()