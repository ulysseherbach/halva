"""Get the precision matrix of a multivariate ordered-probit model."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, truncnorm


# To be improved ###################################################
def preprocess(data: pd.DataFrame):
    """Process data for minimal ordinal structure."""
    n, p = data.shape
    # Check that all samples contain data
    for i in range(n):
        if data.iloc[i].isna().to_numpy().all():
            msg = 'At least one sample does not contain any data'
            raise ValueError(msg)
    # Process data for minimal ordinal structure
    for v in data.columns:
        if data[v].min() > 0:
            data[v] -= data[v].min()
    # Number of possible values for each variable
    data_max = (data.max() + 1).replace(np.nan, 0)
    m = [int(data_max[v]) for v in data.columns]
    # Reference list of all (variable, value) pairs
    ref = [(j, k) for j in range(p) for k in range(m[j])]
    return m, ref
####################################################################


def infer_precision(data: pd.DataFrame, **kwargs):  # noqa: PLR0912, PLR0914, PLR0915
    """Get the precision matrix of a multivariate ordered-probit model.

    data: pandas DataFrame
    """
    edges = kwargs.get('edges')
    n_iter = kwargs.get('n_iter')
    normalize = kwargs.get('norm')
    traj = kwargs.get('traj', False)
    save = kwargs.get('save', False)

    # Number of samples and number of variables
    n, p = data.shape

    # Mask of existing edges
    edge_mask = build_edge_mask(p, edges)

    # Number of possible values for each variable
    m, ref = preprocess(data)

    # Array version
    x = data.to_numpy()

    # Mask of hidden variables (missing data and latent factors)
    h = np.isnan(x)

    # Indices of missing data
    mis = [np.nonzero(h[i])[0] for i in range(n)]

    # Indices of observed data
    obs = [np.nonzero(np.logical_not(h[i]))[0] for i in range(n)]

    # Initialization
    print('Initialization...')

    # Empirical probabilities
    prob = []  # prob[j][k+1] = probability that variable j takes value k
    for j, v in enumerate(data.columns):
        # Case of observed variable
        if m[j] > 0:
            val = data[v][data[v].notna()]
            hist = np.zeros(m[j] + 1)
            hist[1:] = np.histogram(val, bins=m[j])[0]
            prob.append(hist / np.sum(hist))
        # Case of latent variable
        else:
            prob.append(np.array([]))

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
        # Check for numerical issues
        if np.isnan(m1_ref[r]) or np.isnan(m2_ref[r]):
            # Trivial case
            if tau0[r] == tau1[r]:
                msg = 'Warning: some consecutive thresholds are identical'
                print(msg)
                m1_ref[r] = tau0[r]
                m2_ref[r] = tau0[r]**2
            else:
                msg = 'Numerical issue with truncnorm'
                raise ValueError(msg)
        # Values for observed data
        i_obs = (x[:, j] == k)
        m1[i_obs, j] = m1_ref[r]
        m2[i_obs, j] = m2_ref[r]

    # Precision matrix
    theta = np.eye(p)
    theta = -0.001 * np.abs(np.random.standard_normal(size=(p, p)))
    theta += np.eye(p) - np.diag(np.diag(theta))
    theta = (theta + theta.T)/2

    if edges is not None:
        # Model matrix
        model = np.zeros((p, p), dtype=int)
        for u, v in edges:
            model[u, v] = model[v, u] = 1
        # Initialize manifest edges
        for u, v in edges:
            c1, c2 = np.sum(model[u] != 0), np.sum(model[v] != 0)
            if c1 == 1 or c2 == 1:
                theta[u, v] = theta[v, u] = -0.1

    # Export theta trajectory
    if traj:
        traj_theta = np.zeros((p, p, n_iter+1))
        traj_theta[:, :, 0] = theta

    # Individual covariances
    s_ind = np.zeros((n, p, p))

    # Objective function
    q = np.linalg.slogdet(theta)[1] - np.linalg.trace(theta)
    traj_obj = [q]

    # Variational entropy
    ent_ind = np.zeros(n)

    # Main EM loop
    for k in range(n_iter):
        print(f'EM step {k + 1}...')

        # E step
        for i in range(n):
            if np.any(h[i]):

                # Submatrices of theta and m1
                theta_mis = theta[np.ix_(mis[i], mis[i])]
                theta_mis_obs = theta[np.ix_(mis[i], obs[i])]
                m1_obs = m1[[i]][:, obs[i]].T

                # Compute variational parameters
                b = -theta_mis_obs @ m1_obs
                mu = np.linalg.solve(theta_mis, b)[:, 0]
                nu = 1 / np.diag(theta_mis)

                # Update moments
                m1[i, h[i]] = mu
                m2[i, h[i]] = mu**2 + nu

        # Variational covariance matrix
        for i in range(n):
            s_ind[i] = m1[[i]] * m1[[i]].T
            np.fill_diagonal(s_ind[i], m2[i])
        s = np.mean(s_ind, axis=0)

        # Test
        if normalize:
            np.fill_diagonal(s, 1)

        # Variational entropy
        for i in range(n):
            if np.any(h[i]):
                nu = m2[i, h[i]] - m1[i, h[i]]**2
                ent_ind[i] = np.sum(np.log(nu))
        ent = np.mean(ent_ind)

        # M step
        if edge_mask is None:
            # Option 1: unconstrained
            theta[:] = np.linalg.inv(s)
        else:
            # Option 2: constrained structure
            theta[:] = m_step_constrained(s, edge_mask)

        # Record objective function
        q = np.linalg.slogdet(theta)[1] - np.linalg.trace(s @ theta) + ent
        traj_obj.append(q)

        # # Check
        # print(q)

        if traj:
            traj_theta[:, :, k+1] = theta

        if save and (k+1) % 10 == 0:
            fig = plt.figure(figsize=(8, 3))
            x = np.arange(len(traj_obj))
            plt.plot(x, traj_obj, color='r')
            plt.xlim(x[0], x[-1])
            plt.xlabel('Iteration')
            plt.ylabel('Objective')
            fig.savefig(f'traj_obj_{k+1}.pdf', bbox_inches='tight')
            np.save(f'precision_c_v1_{k+1}', theta)

    # Export results
    traj_obj = (n/2) * np.array(traj_obj)
    return (theta, traj_obj, traj_theta) if traj else (theta, traj_obj)


def m_step_constrained(s, edge_mask, **kwargs):
    """
    M step with constrained structure.

    Reference: Algorithm 17.1 in Hastie, Tibshirani & Friedman (2009),
    The Elements of Statistical Learning.
    """
    tol = kwargs.get('tol', 1e-10)
    max_iter = kwargs.get('max_iter', 100)
    p = s.shape[0]
    # Create partition grid
    partition = 1 - np.eye(p, dtype=int)
    # 1. Initialization
    w = s.copy()
    beta = np.zeros((p, p))
    diff = tol + 1
    count = 0
    # 2. Coupled regressions
    while (diff > tol) and (count < max_iter):
        w_old = w.copy()
        for j in range(p):
            # Current partition
            ind = np.nonzero(partition[j])[0]
            part11 = np.ix_(ind, ind)  # All but the jth row and column
            part12 = np.ix_(ind, [j])  # jth column without jth row
            # Remove missing edges
            ind = np.nonzero(edge_mask[j])[0]
            part11r = np.ix_(ind, ind)  # Reduced matrix
            part12r = np.ix_(ind, [j])  # Reduced column
            # Solve reduced regression
            beta[part12r] = np.linalg.solve(w[part11r], s[part12r])
            # Update estimated covariance
            w[part12] = w[part11] @ beta[part12]
            w[j] = w[:, j]  # Ensure symmetry
        diff = np.mean(np.abs(w - w_old))
        count += 1
    # print(f'converged in {count} iterations')
    # 3. Retrieve theta
    theta = np.zeros((p, p))
    for j in range(p):
        ind = np.nonzero(partition[j])[0]
        part12 = np.ix_(ind, [j])  # jth column without jth row
        theta[j, j] = (s[j, j] - w[part12].T @ beta[part12])**(-1)
        theta[part12] = -theta[j, j] * beta[part12]
    # Checks
    # print(np.allclose(theta, np.linalg.inv(w)))
    # print(s, '\n\n', w, '\n\n', theta)
    # print(theta, '\n')
    return theta


def build_edge_mask(p, edges):
    """Mask of existing edges."""
    if edges is None or len(edges) == p * (p-1) / 2:
        edge_mask = None
    else:
        edge_mask = np.zeros((p, p), dtype=int)
        for i, j in edges:
            if i != j:
                edge_mask[i, j] = 1
                edge_mask[j, i] = 1
    return edge_mask
