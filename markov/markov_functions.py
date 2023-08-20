# Functions used to simulate Markov chains

import numpy as np
import quantecon as qe


def mc_sample_path(P, ψ0=None, ts_length=1_000):

    # set up
    P = np.asarray(P)
    X = np.empty(ts_length, dtype=int)

    # Convert each row of P into a cdf
    P_dist = np.cumsum(P, axis=1)  # Convert rows into cdfs
    # draw initial state, defaulting to 0
    if ψ0 is not None:
        # Check that ψ0 is a valid probability mass function
        ψ0 = np.asarray(ψ0)
        pmf_condition = np.all(ψ0 >= 0) and np.abs(ψ0.sum() - 1.0) < 1e-6
        if pmf_condition is False:
            raise ValueError('Invalid ψ0: it must be a probability mass function')
        # draw initial state
        X_0 = qe.random.draw(np.cumsum(ψ0))
    else:
        X_0 = 0

    # simulate
    X[0] = X_0
    for t in range(ts_length - 1):
        X[t+1] = qe.random.draw(P_dist[X[t], :])

    return X


def mc_stationary_distribution(P, initial_guess=None, max_iter=500, tol=1e-6):
    """Stationary distribution of a Markov chain

    :param P: Transition matrix
    :type P: ndarray
    :param initial_guess: Initial guess for the stationary distribution, defaults to None
    :type initial_guess: ndarray, optional
    :param max_iter: Maximum number of iterations, defaults to 500
    :type max_iter: int, optional
    :param tol: Error tolerance, defaults to 1e-6
    :type tol: float, optional
    :return: The stationary distribution of P calculated using successive aproximations
    :rtype: ndarray
    """
    P = np.asarray(P)

    ψ = initial_guess if initial_guess is not None else (np.ones(len(P)) / len(P))

    for i in range(max_iter):
        ψ_new = ψ @ P
        if np.max(np.abs(ψ_new - ψ)) < tol:
            return ψ_new
        else:
            ψ = ψ_new
    else:
        raise ValueError(f'Convergence not reached in {max_iter} iterations')

    