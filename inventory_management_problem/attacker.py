# MIT License (check the LICENSE file)
# Copyright (c) Author of "Balancing detectability and performance 
# of attacks on the control channel of Markov Decision Processes", 
# Alessio Russo, alessior@kth.se, 2021.
#

import numpy as np
from inventory import Inventory
from utils import *
from typing import Callable, List, Tuple
from scipy.special import rel_entr
import cvxpy as cp


def compute_deterministic_attack(gamma, beta, P, R, pi, dfunc, alpha=None):
    ''' Computes a deterministic stealthy attack
        
        Parameters
        -----------
        gamma: float
            Discount factor of the attack
        beta: float
            Penalty factor beta, should be >= 0
        P: np.ndarray
            Probability transition matrix of size |S|x|A|x|S|
        R: np.ndarray
            Reward transition matrix, of size |S|x|A|
        pi: np.ndarray
            Policy of the main agent, of size |S|x|A|
        dfunc: Callable[[int, int], Bool]
            Function that accepts two actions as input and returns True
            if the two actions are sufficiently close, according to some metric
            function
        alpha: np.ndarray
            Probability distribution of the initial state

        Returns
        -----------
        v: float
            Discounted average value
        phi: np.ndarray
            Policy of the attack
        mu: np.ndarray
            Discounted stationary distribution of the attack
     '''
    dim_state, dim_action = P.shape[0], P.shape[1]
    v = cp.Variable((dim_state))
    if not alpha:
        alpha = np.asarray([1 / dim_state] * dim_state)
    else:
        alpha = np.asarray(alpha).flatten()
        assert np.isclose(alpha.sum(), 1.)

    objective = cp.Minimize(cp.sum(cp.multiply(v, alpha)))
    constraints = []
    det_pi = pi.argmax(-1)
    for s in range(dim_state):
        for a in range(dim_action):
            if check_absolute_continuity(P[s,a], P[s,det_pi[s]]) or a==det_pi[s]:
                if (dfunc and dfunc(a, det_pi[s])) or not dfunc:
                    constraints.append(
                        v[s] - (
                            -beta * rel_entr(P[s,a], P[s,det_pi[s]]).sum() 
                            + P[s, a] @ (R[s,a] + gamma * v)
                        ) >= 1e-9)


    problem = cp.Problem(objective, constraints)
    result = problem.solve(verbose=False)
    v = v.value
    x = np.zeros((dim_state, dim_action))
    for s in range(dim_state):
        for a in range(dim_action):
            if check_absolute_continuity(P[s,a], P[s,det_pi[s]]) or a==det_pi[s]:
                if (dfunc and dfunc(a, det_pi[s])) or not dfunc:
                    x[s,a] = -beta * rel_entr(P[s,a], P[s,det_pi[s]]).sum() + P[s, a] @ (R[s,a] + gamma * v)
                else:
                    x[s,a] = -np.infty
            else:
                x[s,a] = -np.infty

    x = x.argmax(-1)
    phi = np.zeros((dim_state, dim_action))
    phi[np.arange(x.size), x] = 1
    return v, phi, compute_discounted_stationary_distribution(gamma, P, phi, alpha)


def compute_randomized_attack(gamma, epsilon, P, R, pi, alpha=None):
    ''' Computes a randomized stealthy attack
        
        Parameters
        -----------
        gamma: float
            Discount factor of the attack
        epsilon: float
            Constraint value, should be >= 0
        P: np.ndarray
            Probability transition matrix of size |S|x|A|x|S|
        R: np.ndarray
            Reward transition matrix, of size |S|x|A|
        pi: np.ndarray
            Policy of the main agent, of size |S|x|A|
        alpha: np.ndarray
            Probability distribution of the initial state

        Returns
        -----------
        result: float
            Discounted average value
        phi: np.ndarray
            Policy of the attack
        mu: np.ndarray
            Discounted stationary distribution of the attack
     '''
    #Construct the problem to find minimum privacy
    dim_state, dim_action = pi.shape
    xi = cp.Variable((dim_state, dim_action), nonneg=True)
    if not alpha:
        alpha = np.asarray([1 / dim_state] * dim_state)
    else:
        alpha = np.asarray(alpha).flatten()
        assert np.isclose(alpha.sum(), 1.)
    det_pi = pi.argmax(-1)

    objective = 0.
    constraints = []
    for s in range(dim_state):
        for a in range(dim_action):
            objective += xi[s,a] *  (P[s,a] @ R[s,a])
            if not check_absolute_continuity(P[s, a], P[s, det_pi[s]]):
                constraints.append(xi[s,a] == 0)
    objective = cp.Maximize(objective / (1 - gamma))

    # stationarity_constraint
    for s in range(dim_state):
        stationarity_constraint = 0
        for a in range(dim_action):
            stationarity_constraint += cp.sum((xi[:, a] @ P[:, a, s]))
        constraints.append(cp.sum(xi[s]) == (1-gamma) * alpha[s] + gamma *  stationarity_constraint)
        
    # Information constraint
    info_constraint = 0
    for s in range(dim_state):
        for a in range(dim_action):
            if check_absolute_continuity(P[s, a], P[s, det_pi[s]]):
                info_constraint += xi[s,a] * rel_entr(P[s, a], P[s, det_pi[s]]).sum()
    constraints.append(info_constraint <= epsilon)


    problem = cp.Problem(objective, constraints)
    result = problem.solve(verbose=False)
    

    xi = xi.value

    for s in range(dim_state):
        for a in range(dim_action):
            if not check_absolute_continuity(P[s, a], P[s, det_pi[s]]):
                xi[s,a] = 0

    phi = np.asarray([xi[s] / xi[s].sum() if xi[s].sum() > 0 else np.ones(dim_state) / dim_state for s in range(dim_state)])

    mu = [xi[s].sum() / xi.sum() if xi[s].sum() > 0  else 0 for s in range(dim_state)]

    return result, phi, mu


if __name__ == '__main__':
    ''' Test code '''
    inventory_size = 30
    gamma = 0.99
    epsilon = 5e-2
    env = Inventory(inventory_size)
    dfunc = lambda x, y: np.abs(x-y) < 3
    V, pi = value_iteration(env, gamma)
    res, phi, mud = compute_randomized_attack(gamma, epsilon, env.P, -env.R, pi)
    mu = compute_stationary_distribution(env.P, phi)
    print(compute_I(mu, env.P, pi, phi))
    print(compute_I_gamma(mud, env.P, pi, phi))
    print(V.mean())
    print(res)
