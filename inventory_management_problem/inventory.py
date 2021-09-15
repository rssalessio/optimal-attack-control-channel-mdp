# MIT License (check the LICENSE file)
# Copyright (c) Author of "Balancing detectability and performance 
# of attacks on the control channel of Markov Decision Processes", 
# Alessio Russo, alessior@kth.se, 2021.
#

import numpy as np
import scipy.stats as stats

class Inventory(object):
    ''' Inventory class, used to model the inventory management problem 
        
        Parameters
        -----------
        inventory_size: int
            Size of the inventory
        fixed_cost: int
            Fixed cost of purchasing items
        item_cost: int
            Unitary cost of an item
        holding_cost: int
            Cost of holding an item
        item_price: int
            Selling price of an item
        order_rate: int
            Items order rate (parameter of the Poisson distribution)
        seed: int
            Seed used to initialize numpy
    '''
    def __init__(self, 
                 inventory_size: int = 50,
                 fixed_cost: int = 3,
                 item_cost: int = 2,
                 holding_cost: int = 2,
                 item_price: int = 4,
                 order_rate: int = 6,
                 seed: int = None):
        self.inventory_size = inventory_size
        self.fixed_cost = fixed_cost
        self.item_cost = item_cost
        self.holding_cost = holding_cost
        self.item_price = item_price
        self.order_rate = order_rate

        assert item_price > holding_cost

        np.random.seed(seed)
        self.compute_P_matrix()
        self.compute_R_matrix()
        self.reset()

    def transition(self, state: int, action: int, demand: int) -> int:
        return max(min(state + action, self.inventory_size) - demand, 0)

    def compute_transition_probability(self, state: int, action: int, next_state: int) -> float:
        return self.P[state, action, next_state]

    def compute_P_matrix(self):
        ''' Computes the transition probability matrix '''
        # S x A x D x S
        D = self.inventory_size + 1
        self.P = np.zeros((self.inventory_size + 1, self.inventory_size + 1, self.inventory_size + 1))
        _max = np.broadcast_to([self.inventory_size + 1], (self.inventory_size + 1))
        _zeros = np.zeros(self.inventory_size + 1)
        yvec = np.arange(self.inventory_size + 1)

        for s in range(self.inventory_size + 1):
            svec = np.broadcast_to([s], (self.inventory_size + 1))
            for a in range(self.inventory_size + 1):
                avec = np.broadcast_to([a], (self.inventory_size + 1))
                for d in range(D):
                    prior = stats.poisson.pmf(d, self.order_rate)
                    dvec = np.broadcast_to([d], (self.inventory_size + 1))
                    self.P[s,a,:] += (yvec == np.max([np.min([svec + avec, _max], 0) - dvec, _zeros], 0)) * prior
                self.P[s,a,:] /= (1e-9 + np.sum(self.P[s,a,:]))

        return self.P

    def compute_R_matrix(self):
        self.R = np.zeros((self.inventory_size + 1, self.inventory_size + 1, self.inventory_size + 1))
        for s in range(self.inventory_size + 1):
            for a in range(self.inventory_size + 1):
                for y in range(self.inventory_size + 1):
                    self.R[s,a,y] = self.reward(s, a, y)
        return self.R


    def reward(self, state: int, action: int, next_state: int) -> float:
        ''' Computes the reward given current state, action and next state '''
        fixed_cost = -self.fixed_cost * (action > 0)
        holding_cost = -self.holding_cost * state
        item_cost = -self.item_cost * max(min(state + action, self.inventory_size) - state, 0)
        sellings = self.item_price * max(min(state + action, self.inventory_size) - next_state, 0)
        return fixed_cost + holding_cost + item_cost + sellings

    def step(self, action: int) -> (int, float):
        ''' Takes an action and returns next state and reward '''
        action = int(action)
        assert action >= 0 and action <= self.inventory_size

        demand =  np.random.poisson(self.order_rate)
        next_state = self.transition(self.state, action, demand)
        reward = self.reward(self.state, action, next_state)
        self.state = next_state
        return self.state, reward

    def reset(self) -> int:
        ''' Resets the environment to the initial state '''
        self.state = self.inventory_size
        return self.state