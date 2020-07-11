"""
    Simulation Run
"""
from collections import defaultdict

import numpy as np

from utils import get_ucb_val

class SimulationRun():
    """
    Simulation Run
    """
    def __init__(self, K, T, sequence=False):
        self.K = K
        self.T = T
        self.counters = [0 for _ in range(K)]
        self.total_rewards = [0 for _ in range(K)]
        self.expected_rewards = [1 for _ in range(K)]
        self.t = 0
        self.ucb_vals = [np.infty for _ in range(K)]
        self.intervals = []

        if sequence:
            self.sequence = []
            self.rewards = defaultdict(list)
        else:
            self.sequence = None
        self.last_pulls = [0 for _ in range(K)]
        self.delta_approx = None
        self.f_approx = None
        self.alg_rounds = None
        self.opportunities_rounds = None
        self.utility = None

    def update_arm(self, i, reward):
        """Update counter, total reward, expected reward
        ucb value of a pulled arm
        """
        self.counters[i] += 1
        self.total_rewards[i] += reward
        if self.sequence is not None:
            self.sequence.append(i)
            self.rewards[i].append(reward)
        self.last_pulls[i] = self.t
        self.expected_rewards[i] = self.total_rewards[i]/self.counters[i]
        self.t += 1
        self.ucb_vals[i] = min(
            1, get_ucb_val(self.expected_rewards[i], self.counters[i], self.T))

    def get_total_reward(self):
        """Get the total obtained reward
        """
        return sum(self.total_rewards)

    def set_delta_approx(self, t):
        """Set the number of rounds delta was approximated
        """
        self.delta_approx = t

    def set_f_approx(self, t):
        """Set the number of rounds f was approximated
        """

        self.f_approx = t

    def set_alg_rounds(self, t):
        """Set the number of rounds alg was invoked
        """
        self.alg_rounds = t

    def set_opportunities_rounds(self, t):
        """Set the number of opprotunities rounds
        """
        self.opportunities_rounds = t

    def set_utility(self, utility):
        """Set the obtained utility
        """
        self.utility = utility

    def set_ti(self, ti):
        self.ti = ti

    def add_interval(self, interval):
        self.intervals.append(interval)

    def __str__(self):
        return (f"N: {self.counters}, "
                f"mu: {self.expected_rewards}, "
                f"reward: {sum(self.total_rewards)}")
