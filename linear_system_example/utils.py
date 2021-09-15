# MIT License (check the LICENSE file)
# Copyright (c) Author of "Balancing detectability and performance 
# of attacks on the control channel of Markov Decision Processes", 
# Alessio Russo, alessior@kth.se, 2021.
#

import cvxpy as cp
import numpy as np
from scipy.linalg import solve_discrete_lyapunov, solve_discrete_are
from collections import namedtuple
System = namedtuple('System', ['A', 'B', 'K', 'Kbar', 'R', 'theta', 'Q'])


def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    ''' Compute optimal feedback gain for a discrete-time system '''
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.inv(B.T @ P @ B + R) * (B.T @ P @ A)

def compute_power_series(X: np.ndarray, Z: np.ndarray = None, eps: float = 1e-5):
    ''' Function used to compute Y=\\sum_{i=0}^\\infty (X^T)^i Z X^i 
        eps tunes the approximated value of the series.
        Returns Y and the value of i for which the series is truncated.
    '''
    L_bar = np.identity(X.shape[0])
    L_pwr = np.matrix(np.identity(X.shape[0]))
    n = 0
    if Z is None:
        Z = np.matrix(np.identity(X.shape[0]))
    elif np.all(np.isclose(Z, 0.)):
        return 0*X, 0, 0
    while np.linalg.norm(L_pwr) > eps:
        L_pwr = L_pwr @ (X.T @ Z @ X)
        L_bar += L_pwr
        n += 1
    return L_bar, L_pwr, n

def compute_R(sys: System, beta: float, Ebar: np.ndarray) -> np.ndarray:
    ''' Computes the optimal attack covariance matrix according to the
        value of beta, Ebar, Kbar
    '''
    Binv = np.linalg.inv(sys.B.T @ sys.B) @ sys.B.T @ sys.Q
    Qinv = np.linalg.inv(sys.Q)
    E1 = 0.5 * beta * compute_power_series(Ebar, sys.Kbar.T @ sys.B.T @ Qinv @ sys.B @ sys.Kbar)[0]
    E2 = compute_power_series(Ebar)[0]

    Ebarinv = np.linalg.inv(E1 -E2)
    R = - Binv @ np.linalg.inv(0.5 * beta * Ebarinv + sys.Q) @ Binv.T
    return R

def compute_llr(x: float,  next_x: float, sys: System) -> float:
    ''' Computes log-likelihood ratio '''
    n,m = sys.B.shape
    mu0 = (sys.A + sys.B @ sys.K) @ x
    mu1 = mu0 + sys.B @ sys.Kbar @ x
    Sigma1 = sys.Q + sys.B @ sys.R @ sys.B.T
    Sigma0 = sys.Q
    z = next_x[None, :]

    t1 = -0.5 * (z- mu1) @ np.linalg.inv(Sigma1) @ (z - mu1).T
    t1 -= 0.5 * np.log(np.linalg.det(Sigma1))
    t0 = -0.5 * (z - mu0) @ np.linalg.inv(Sigma0) @ (z - mu0).T
    t0 -= 0.5 * np.log(np.linalg.det(Sigma0))

    return (t1 - t0).item()


def simulate(N: int, T: int, sys: System, attack_time: int = 0) -> (np.ndarray, np.ndarray):
    ''' Performs N simulations of the given system, each of T steps 
        returns the states and actions
    '''
    n, m = sys.B.shape
    x = np.zeros((N, T+1, n))
    u = np.zeros((N, T, m))
    llr =  np.zeros((N, T))
    cusum = np.zeros((N, T+1))
    for i in range(N):
        x[i, 0] = np.random.normal(size=(n,))
        for t in range(T):
            u[i, t] = sys.K @ x[i, t]
            if t >= attack_time:
                u[i, t] += (sys.Kbar @ x[i, t] + np.random.multivariate_normal(np.zeros(m), sys.R)).tolist()[0]
            # u[i, t, :] = ((sys.K + sys.Kbar) @ x[i, t,:] + np.random.multivariate_normal(np.zeros(m), sys.R)).flatten()
            x[i, t+1,:] = sys.A@x[i, t] + sys.B@u[i, t] + np.random.multivariate_normal(np.zeros(n), sys.Q)
            llr[i, t] = compute_llr(x[i, t], x[i, t+1], sys)
            cusum[i, t+1] = max(0,  cusum[i, t]+llr[i, t]) 
    return x, u, llr, cusum

def compute_beta_theta(sys: System, Linv: np.ndarray) -> float:
    ''' Compute the minimum value of beta for theta '''
    I = np.identity(sys.A.shape[0])
    Qinv = np.linalg.inv(sys.Q)
    x = sys.B.T @ Linv.T @ Linv @ sys.B
    y = sys.B.T @ (I + sys.B @ sys.Kbar @ Linv).T @ Qinv @ (I + sys.B @ sys.Kbar @ Linv) @ sys.B
    return 2 * np.linalg.eig(x)[0].max() / np.linalg.eig(y)[0].min()

def compute_beta_R(sys: System, Enum: np.ndarray, Eden: np.ndarray) -> (float, float):
    ''' Compute the values beta_min, beta_max for the optimal attack
        covariance matrix R
    '''
    beta = cp.Variable(1)
    I = np.identity(sys.A.shape[0])
    Ebeta = (0.5 * beta * Eden - Enum)

    constraints = [-Ebeta >> 1e-9 * I, beta >= 1e-5, 0.5 * beta * I + sys.Q @ Ebeta >> 1e-9*I]
    prob = cp.Problem(cp.Minimize(beta), constraints)
    prob.solve()

    beta_min = beta.value.item()
    prob = cp.Problem(cp.Maximize(beta), constraints)
    prob.solve()
    if beta.value is None:
        beta_max = np.infty
    else:
        beta_max = beta.value.item()
    return beta_min, beta_max

def compute_I(sys: System, Linv: np.ndarray) -> float:
    ''' Compute information rate '''
    I = np.identity(sys.A.shape[0])
    Qinv = np.linalg.inv(sys.Q)
    term1 = np.trace(Qinv @ sys.B @ sys.R @ sys.B.T)
    term2 = np.log(np.linalg.det(sys.Q) / np.linalg.det(sys.Q + sys.B @ sys.R @ sys.B.T))
    x = sys.B.T @ (I +  sys.B @ sys.Kbar @ Linv).T @ Qinv @ (I +  sys.B @ sys.Kbar @ Linv) @ sys.B
    term3 = sys.theta.T @ x @ sys.theta
    Sigma = solve_discrete_lyapunov(sys.A + sys.B @ sys.K + sys.B @ sys.Kbar, sys.Q + sys.B @ sys.R @ sys.B.T)
    term4 = np.trace(sys.Kbar.T @ sys.B.T @ Qinv @ sys.B @ sys.Kbar @ Sigma)
    return 0.5 * (term1 + term2 + term3 + term4)

def compute_attack(sys: System, debug: bool = False):
    ''' Computes R, beta_min, beta_max '''
    # Eigenvalues
    I = np.identity(sys.A.shape[0])
    E = sys.A + sys.B @ sys.K
    Ebar = sys.A + sys.B @ (sys.K + sys.Kbar)
    L = I - Ebar
    Linv = np.linalg.inv(L)
    Qinv = np.linalg.inv(sys.Q)
    eigE = np.linalg.eig(E)[0]
    eigEbar = np.linalg.eig(Ebar)[0]
    if debug:
        print('Eigenvalues of A+BK: {}'.format(eigE))
        print('Eigenvalues of A+BK+BKbar: {}'.format(eigEbar))


    # Check condition on R
    Epower_num = compute_power_series(Ebar)[0]
    Epower_den = compute_power_series(Ebar, sys.Kbar.T @ sys.B.T @ Qinv @ sys.B @ sys.Kbar)[0]

    beta_min, beta_max = compute_beta_R(sys, Epower_num, Epower_den)
    beta_theta = compute_beta_theta(sys, Linv)

    # Compute R
 
    beta = beta_min * (1 + 1e-3)
    R = compute_R(sys, beta, Ebar)
    if debug:
        print('Chosen beta: {} - {} - beta_theta: {}'.format(beta_min, beta_max, beta_theta))
        print('Matrix R: \n{}'.format(R))
        print('Eigenvalues of R: {}'.format(np.linalg.eig(R)[0]))
        
    if beta_max < np.infty:
        beta_min = beta_min + (beta_max-beta_min)/50
    return R, (beta_min, beta_max), beta, Linv, Ebar


def compute_ergodic_reward(sys: System, Linv: np.ndarray) -> float:
    ''' Computes the ergodic reward of a linear system '''
    term1 = sys.theta.T @ (sys.B.T @ Linv.T @ Linv @ sys.B) @ sys.theta
    term2 = np.trace(
            solve_discrete_lyapunov(
                sys.A + sys.B @ sys.K + sys.B @ sys.Kbar, sys.Q + sys.B @ sys.R @ sys.B.T)
            )
    return term1 + term2
