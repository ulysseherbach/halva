import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm

def infer_precision(data: pd.DataFrame, n_iter=100):
    """
    Estimate the precision matrix of a multivariate ordered-probit model.

    data: pandas DataFrame
    """
    # Number of samples and number of variables
    n, p = data.shape

    # To be improved ##########################################################
    # Process data for minimal ordinal structure
    for v in data.columns:
        if data[v].min() > 0:
            data[v] = data[v] - data[v].min()
    # Number of possible values for each variable
    m = [int(data[v].max()) + 1 for v in data.columns]
    ###########################################################################

    # Reference list of all (variable, value) pairs
    ref = [(j, k) for j in range(p) for k in range(m[j])]

    # Array version
    x = data.to_numpy()

    # Mask of hidden variables (missing data and latent factors)
    h = np.isnan(x)

    # Indices of missing data
    mis = [np.nonzero(h[i])[0] for i in range(n)]

    # Indices of observed data
    obs = [np.nonzero(np.logical_not(h[i]))[0] for i in range(n)]

    # Initialization
    print("Initialization...")

    # Empirical probabilities
    prob = [] # prob[j][k+1] = probability that variable j takes value k
    for j, v in enumerate(data.columns):
        val = data[v][data[v].notna()]
        hist = np.zeros(m[j] + 1)
        hist[1:] = np.histogram(val, bins=m[j])[0]
        prob.append(hist / hist.sum())

    # Cumulative distribution functions (lower and upper)
    cdf = [np.cumsum(prob[j]) for j in range(p)]
    cdf0 = [min(c, 1) for j in range(p) for c in cdf[j][:-1]]
    cdf1 = [min(c, 1) for j in range(p) for c in cdf[j][1:]]

    # Estimated thresholds indexed by ref (lower and upper)
    tau0 = norm.ppf(np.array(cdf0))
    tau1 = norm.ppf(np.array(cdf1))

    # Sufficient statistics indexed by ref
    m1_ref = truncnorm.moment(1, tau0, tau1)
    m2_ref = truncnorm.moment(2, tau0, tau1)

    # Moments of order 1 and 2
    m1 = np.zeros((n, p))
    m2 = np.zeros((n, p))

    for r, (j, k) in enumerate(ref):
        # Check for numerial issues
        if np.isnan(m1_ref[r]) or np.isnan(m2_ref[r]):
            # Trivial case
            if tau0[r] == tau1[r]:
                m1_ref[r] = tau0[r]
                m2_ref[r] = tau0[r]**2
            else:
                raise ValueError("Numerical issue with truncnorm")
        # Values for observed data
        else:
            i_obs = (x[:, j] == k)
            m1[i_obs, j] = m1_ref[r]
            m2[i_obs, j] = m2_ref[r]

    # Precision matrix
    theta = np.eye(p)

    # Individual covariances
    s_ind = np.zeros((n, p, p))

    # Objective function
    obj_traj = []

    # Variational entropy
    ent_ind = np.zeros(n)

    # Main EM loop
    for _ in range(n_iter):
        print(f"EM step {_ + 1}...")

        # E step
        for i in range(n):
            if np.any(h[i]):

                # Submatrices of theta and m1
                theta_mis = theta[np.ix_(mis[i], mis[i])]
                theta_mis_obs = theta[np.ix_(mis[i], obs[i])]
                m1_obs = m1[[i]][:, obs[i]].T

                # Compute variational parameters
                b = -theta_mis_obs @ m1_obs
                mu = np.linalg.solve(theta_mis, b)[:,0]
                nu = 1 / np.diag(theta_mis)

                # Update moments
                m1[i, h[i]] = mu
                m2[i, h[i]] = mu**2 + nu

                # Variational entropy (partial)
                ent_ind[i] = 0.5 * np.sum(np.log(nu))

        # M step
        for i in range(n):

            # Option 1 - Unconstrained
            s_ind[i] = m1[[i]].T @ m1[[i]]
            np.fill_diagonal(s_ind[i], m2[i])

            # Option 2 - Constrained (structure + lasso)
            # TODO

        # Build global covariance
        s = np.mean(s_ind, axis=0)

        # Update precision matrix
        theta[:] = np.linalg.inv(s)

        # Record objective function
        ent = np.mean(ent_ind)
        q = np.linalg.slogdet(theta)[1] - np.linalg.trace(s @ theta) + ent
        obj_traj.append(q)

    return theta, np.array(obj_traj)
