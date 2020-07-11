"""Fairness functions
"""
import copy

import numpy as np

from utils import get_confidence_interval


def prone_rewards(rewards):
    """Prone reward to be between 0 to 1s
    """
    if isinstance(rewards, list):
        return [max(min(r, 1), 0) for r in rewards]
    return max(min(rewards, 1), 0)


class TempSoftmax():
    """Softmax function
    """
    def __init__(self, temp, portion=1):
        self.temp = temp
        self.portion = portion

    def get_l(self):
        """Get function lipschitz constant"""
        return 1/self.temp

    def f1(self, mean_rewards, t=1, i=None):
        """Apply function"""
        proned_rewards = prone_rewards(mean_rewards)
        exps = [np.exp(r/self.temp) for r in proned_rewards]
        denominator = sum(exps)
        portions = [(self.portion*t*e)/denominator for e in exps]
        if i is not None:
            return portions[i]
        return portions

    def get_max_diff_arm(self, mean_rewards, counter, T, i):
        """Get the maximal difference in the hypercube for arm i
        and which arms should be pulled in order to improve the approximation
        """
        proned_rewards = prone_rewards(mean_rewards)
        confidence_intervals = [
            get_confidence_interval(cnt, T) for cnt in counter]

        ucb_values = prone_rewards(
            [x+y for x, y in zip(proned_rewards, confidence_intervals)])
        lcb_values = prone_rewards(
            [x-y for x, y in zip(proned_rewards, confidence_intervals)])

        max_vec = copy.deepcopy(lcb_values)
        max_vec[i] = ucb_values[i]
        min_vec = copy.copy(ucb_values)
        min_vec[i] = lcb_values[i]
        max_val = self.f1(max_vec, t=1, i=i)
        min_val = self.f1(min_vec, t=1, i=i)
        return max_val-min_val, range(len(mean_rewards))

    def __str__(self):
        return f"TempSoftmax_temp={self.temp}_p={self.portion}"


class CartesianEqualPortion():
    def __init__(self, p, k):
        self.p = p
        self.k = k

    def f1(self, mean_rewards, t=1, i=None):
        """Apply function"""
        if i is not None:
            return (t * self.p)/self.k
        return [(t * self.p) / self.k for _ in mean_rewards]

    def get_l(self):
        """Get function lipschitz constant"""
        return self.p / self.k

    def get_max_diff_arm(self, mean_rewards, counter, T, i):
        """Get the maximal difference in the hypercube for arm i
        and which arms should be pulled in order to improve the approximation
        """
        return (0, set([i]))

    def __str__(self):
        return f"CartesianEqualPortion_p={self.p}"


class CartesianMuPortion():
    def __init__(self, alpha, k):
        self.alpha = alpha
        self.k = k

    def f1(self, mean_rewards, t=1, i=None):
        """Apply function"""
        proned_rewards = prone_rewards(mean_rewards)
        result = [(t * r * self.alpha) / self.k for r in proned_rewards]
        if i is not None:
            return result[i]
        return result

    def get_l(self):
        """Get function lipschitz constant"""
        return self.alpha / self.k

    def get_max_diff_arm(self, mean_rewards, counter, T, i):
        """Get the maximal difference in the hypercube for arm i
        and which arms should be pulled in order to improve the approximation
        """
        mean_reward = mean_rewards[i]
        confidence_interval = get_confidence_interval(counter[i], T)
        mean_rewards_ucb = copy.copy(mean_rewards)
        mean_rewards_ucb[i] = prone_rewards(mean_reward + confidence_interval)
        mean_rewards_lcb = copy.copy(mean_rewards)
        mean_rewards_lcb[i] = prone_rewards(mean_reward - confidence_interval)
        max_val = self.f1(mean_rewards_ucb, i=i)
        min_val = self.f1(mean_rewards_lcb, i=i)

        return (max_val-min_val, set([i]))

    def __str__(self):
        return f"CartesianMuPortion_alpha={self.alpha}"
