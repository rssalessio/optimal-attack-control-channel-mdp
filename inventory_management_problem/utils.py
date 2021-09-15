# MIT License (check the LICENSE file)
# Copyright (c) Author of "Balancing detectability and performance 
# of attacks on the control channel of Markov Decision Processes", 
# Alessio Russo, alessior@kth.se, 2021.
#

import numpy as np
import cvxpy as cp
from typing import List
from scipy.special import rel_entr

def value_iteration(env, gamma: float, eps: float = 1e-9) -> (np.ndarray, np.ndarray):
    """
    Value iteration
    
    Parameters
    ------------------
        env: Inventory
            Environment
        gamma: float
            Discount factor
        eps: float, optional
            Value iteration tolerance

    Returns
    ------------------
        V: List[float]
            Value function
        pi: List[int]
            Policy
    """
    P, R = env.P, env.R
    dim_state, dim_action = P.shape[0], P.shape[1]
    V = np.zeros((dim_state))
    pi = np.zeros((dim_state, dim_action))
    while True:
        prevV = np.copy(V)
        V = np.sum(P * (R + gamma * V), axis=-1).max(axis=-1)
        if np.abs(prevV - V).max() < eps:
            break
    x = np.sum(P * (R + gamma * V), axis=-1).argmax(axis=-1)
    pi[np.arange(x.size), x] = 1
    return V, pi


def check_absolute_continuity(P: np.ndarray, Q: np.ndarray) -> bool:
    """ Checks P << Q """
    return np.all(np.isclose(P[np.isclose(Q, 0.)], 0.))


def check_absolute_continuity_policies(P: np.ndarray, pi: np.ndarray, phi: np.ndarray) -> bool:
    """ Checks absolute continuity of two policies given an environment 
        Assume pi is deterministic, of dimension SxA
    """
    dim_state, dim_action = P.shape[0], P.shape[1]
    for s in range(dim_state):
        Q = P[s, pi[s].argmax()]
        F = phi[s] @ P[s]
        if not check_absolute_continuity(F, Q):
            return False
    return True


def check_absolute_continuity_policies_strict(P: np.ndarray, pi: np.ndarray, phi: np.ndarray) -> bool:
    """ Strict check of absolute continuity of two policies given an environment
        Used when computing bar I. Assume pi is deterministic, of dimension SxA
    """
    dim_state, dim_action = P.shape[0], P.shape[1]
    for s in range(dim_state):
        Q = P[s, pi[s].argmax()]
        for a in range(dim_action):
            if phi[s,a] >0:
                F = P[s, a]
                if not check_absolute_continuity(F, Q):
                    return False
    return True

def build_markov_transition_density(P: np.ndarray, pi: np.ndarray):
    """ Computes the transition density P^{pi}(x'|x) given a policy pi
    Parameters
    ----------
    P : np.ndarray
        Numpy matrix containing the transition probabilities for the model
        The matrix should have dimensions |S|x|A|x|S|
    pi : np.ndarray
        Numpy matrix of dimensions |S|x|A| containing the
        policy probabilities
    Returns
    -------
    P_pi : np.ndarray
        Transition matrix
    """
    na, ns = P.shape[1], P.shape[0]
    P_pi = np.zeros((ns, ns))
    for s in range(ns):
        for y in range(ns):
            P_pi[y, s] = np.dot(P[s, :, y], pi[s, :])
    return P_pi

def compute_stationary_distribution(
        P: np.ndarray, pi: np.ndarray) -> (np.ndarray):
    """ Computes stationary distribution given the transition density matrix
        and the policy.
    Parameters
    ----------
    P : np.ndarray
        Numpy matrix containing the transition probabilities for the model
        The matrix should have dimensions |S|x|A|x|S|
    pi : np.ndarray
        Numpy matrix of dimensions |S|x|A| containing the
        policy probabilities
    Returns
    -------
    mu : np.ndarray
        Stationary state distribution
    """
    na, ns = P.shape[1], P.shape[0]
    P_pi = build_markov_transition_density(P, pi)

    eigs, u = np.linalg.eig(P_pi)
    # 0 should be the index of the eigenvalue 1
    mu = np.abs(u[:, 0]) / np.sum(np.abs(u[:, 0]))
    assert np.isclose(eigs[0], 1.)
    assert np.isclose(np.sum(mu), 1.)
    return mu

def compute_discounted_stationary_distribution(
        gamma: float, P: np.ndarray, pi: np.ndarray, alpha: np.ndarray) -> (np.ndarray):
    """ Computes stationary distribution given the transition density matrix
        and the policy.
    Parameters
    ----------
    gamma: float
        Discount factor
    P : np.ndarray
        Numpy matrix containing the transition probabilities for the model
        The matrix should have dimensions |S|x|A|x|S|
    pi : np.ndarray
        Numpy matrix of dimensions |S|x|A| containing the
        policy probabilities
    alpha: np.ndarray
        Initial state distribution
    Returns
    -------
    mu : np.ndarray
        Discoutned  stationary state distribution
    """
    dim_state, dim_action = P.shape[0], P.shape[1]
    mu = cp.Variable((dim_state, dim_action), nonneg=True)

    alpha = np.asarray(alpha).flatten()
    assert np.isclose(alpha.sum(), 1.)
    constraints = []

    # Find discounted stationary distribution
    for s in range(dim_state):
        for a in range(dim_action):
            constraints.append(mu[s,a] == pi[s,a] * cp.sum(mu[s]))

    objective = cp.Maximize(1)

    # stationarity_constraint
    for s in range(dim_state):
        stationarity_constraint = 0
        for a in range(dim_action):
            stationarity_constraint += cp.sum((mu[:, a] @ P[:, a, s]))
        constraints.append(cp.sum(mu[s]) == (1-gamma) * alpha[s] + gamma *  stationarity_constraint)

    problem = cp.Problem(objective, constraints)
    result = problem.solve(verbose=False)
    return mu.value.sum(axis=-1)


def compute_I(mu: np.ndarray, P: np.ndarray, pi: np.ndarray, phi: np.ndarray) -> (float, float):
    ''' Returns I and \bar I 

        Parameters
        ----------
        mu: np.ndarray
            Ergodic stationary distribution of size |S|
        P:  np.ndarray
            Numpy matrix containing the transition probabilities for the model
            The matrix should have dimensions |S|x|A|x|S|
        pi: np.ndarray
            Numpy matrix of dimensions |S|x|A| containing the
            policy probabilities
        phi: np.ndarray
            Adversarial policy, of size |S|x|A|
        Returns
        -------
        I :    float
            Information rate
        Ibar : float
            Upper bound on the information rate
    '''
    dim_state, dim_action = phi.shape
    I = 0.
    upper_bound = 0.
    for s in range(dim_state):
        if np.isclose(mu[s], 0.):
            continue
        original_action = pi[s].argmax()
        Q = P[s, original_action]
        F = phi[s] @ P[s]
        F[np.isclose(F, 0)] = 0
        #if check_absolute_continuity(F, Q):
        I += mu[s] * rel_entr(F, Q).sum()
        for a in range(dim_action):
            F = P[s, a]
            F[np.isclose(F, 0)] = 0
            if not np.isclose(phi[s,a], 0.) :
                if rel_entr(F, Q).sum()==np.infty or rel_entr(F, Q).sum()==np.nan:
                    import pdb
                upper_bound += mu[s] * phi[s, a] * rel_entr(F, Q).sum()
    return I, upper_bound


def compute_I_gamma(mu: np.ndarray, P: np.ndarray, pi: np.ndarray, phi: np.ndarray) -> float:
    ''' Returns Igamma  and \bar Igamma 

        Parameters
        ----------
        mu: np.ndarray
            Discounted stationary distribution of size |S|
        P:  np.ndarray
            Numpy matrix containing the transition probabilities for the model
            The matrix should have dimensions |S|x|A|x|S|
        pi: np.ndarray
            Numpy matrix of dimensions |S|x|A| containing the
            policy probabilities
        phi: np.ndarray
            Adversarial policy, of size |S|x|A|
        Returns
        -------
        Igamma :    float
            Discounted Information rate
        bar_I_gamma : float
            Upper bound on the discounted information rate
    '''
    dim_state, dim_action = P.shape[0], P.shape[1]
    I_gamma = 0
    bar_I_gamma = 0
    for s in range(dim_state):
        if np.isclose(mu[s], 0.):
            continue
        original_action = pi[s].argmax()
        Q = P[s, original_action]
        F = phi[s] @ P[s]
        F[np.isclose(F, 0)] = 0
        #if check_absolute_continuity(F, Q):
        I_gamma += mu[s] * rel_entr(F, Q).sum()
        for a in range(dim_action):
            F = P[s, a]
            F[np.isclose(F, 0)] = 0
            if not np.isclose(phi[s,a], 0.):
                bar_I_gamma += mu[s] * phi[s, a] * rel_entr(F, Q).sum()

    return I_gamma, bar_I_gamma



if __name__ == '__main__':
    ''' Test code '''
    from inventory import Inventory
    inventory_size = 15
    gamma = 0.95
    env = Inventory(inventory_size)
    V, pi = value_iteration(env, gamma, 1e-6)
    print('----- Value -----')
    print(V)
    print('\n----- Policy -----')
    print(pi.argmax(-1))
    print('\n----- Stationary distribution -----')
    mu = compute_stationary_distribution(env.P, pi)
    print([np.round(mu[s], 2) for s in range(env.P.shape[0])])
    print('\n----- Stationary discounted distribution -----')
    mud = compute_discounted_stationary_distribution(gamma, env.P, pi)
    print([np.round(mud[s], 2) for s in range(env.P.shape[0])])
    print('\n----- Compute I if phi=pi -----')
    print(compute_I(mu, env.P, pi, pi))
    print('\n----- Compute Igamma if phi=pi -----')
    print(compute_I_gamma(mud, env.P, pi, pi))
