"""
Utilities model
"""
import numpy as np

def get_ucb_val(expected_reward, counter, T):
    """Return upper bound of expected reward
    """
    return expected_reward + get_confidence_interval(counter, T)


def get_confidence_interval(counter, T):
    """Retrun confidence interval size - sqrt(2log T/counter)
    """
    if counter == 0:
        return np.infty
    return np.sqrt(2 * np.log(T) / counter)
