import numpy as np

def posterior(X, y, m0, S0, sigma2):

    s0_inv = np.linalg.inv(S0)
    sn =  np.linalg.inv(s0_inv+ (1/sigma2) * X.T @ X)
    sn1 = s0_inv @ m0 + (1/sigma2) *X.T @ y
    mn = sn@sn1 

    return mn, sn
 

def sequential_posterior(X, y, m0, S0, sigma2):
    # return (mN, SN, means)

    m = m0.copy()
    S = S0.copy()

    means = []

    for i in range(len(X)):
        m, S = posterior(
            X[i:i+1],
            y[i:i+1],
            m,
            S,
            sigma2)

        means.append(m.copy())


    return m, S, np.array(means)
   
def predictive(X_test, mN, SN, sigma2):
    p_mean = X_test @ mN
    p_var = sigma2 + np.sum((X_test @ SN)*X_test, axis=1)
    
    return (p_mean, p_var)

