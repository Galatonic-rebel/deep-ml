import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    # Your code here
    m,n = X.shape
    y= y.reshape(-1,1)
    theta = np.zeros((n,1))

    if method == 'batch':
        for i in range(n_epochs):
            prediction = X @ theta
            error = prediction - y
            gradient = (2/m) * X.T @ error
            theta = theta - learning_rate*gradient
        
        return theta.flatten()

    elif method == 'stochastic':
        for i in range(n_epochs):
            for j in range(m):
                prediction = X[j:j+1] @ theta
                error = prediction - y[j]
                gradient = 2*X[j:j+1].T @ error
                theta = theta - learning_rate*gradient
        
        return theta.flatten()

    else:
        for i in range(n_epochs):
            for start in range(0, m, batch_size):
                end = start + batch_size

                X_batch = X[start:end]
                y_batch = y[start:end]
                prediction = X_batch @ theta
                error = prediction - y_batch
                gradient = (2/len(X_batch)) * X_batch.T @ error
                theta = theta - learning_rate*gradient
        
        return theta.flatten()

