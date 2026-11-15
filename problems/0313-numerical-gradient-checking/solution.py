import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    # Your code here
    numerical_grad = np.zeros(len(x))

    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()

        x1[i] += epsilon
        x2[i] -= epsilon

        numerical_grad[i] = (f(x1)- f(x2))/(2*epsilon)

    relative_error= (np.linalg.norm(numerical_grad-analytical_grad)/(np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad)))
    

    return numerical_grad, relative_error
    