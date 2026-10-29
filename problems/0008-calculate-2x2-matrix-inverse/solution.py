def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    """
    Calculate the inverse of a 2x2 matrix.
    
    Args:
        matrix: A 2x2 matrix represented as [[a, b], [c, d]]
    
    Returns:
        The inverse matrix as a 2x2 list, or None if the matrix is singular
        (i.e., determinant equals zero)
    """
    # Your code here
    a,b = matrix[0]
    c,d = matrix[1]

    det = (a*d) - (b*c)
    if det == 0:
        return None
    else:
        inverse = [[d, -b], [-c, a]]
        det_inverse =  [
            [d * (1/det), -b * (1/det)],
            [-c * (1/det), a * (1/det)]
        ]
        return det_inverse

