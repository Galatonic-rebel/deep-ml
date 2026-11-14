import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    # Your code here, make sure to round
    import numpy as np
    X = np.array(X)
    y = np.array(y)

    a = np.linalg.inv(np.dot(X.T, X))
    b = np.dot(X.T, y)
    theta= np.dot(a,b)


    return np.round(theta,4)