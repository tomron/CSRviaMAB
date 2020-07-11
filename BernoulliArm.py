"""BernoulliArm"""

import scipy.stats


class BernoulliArm():
    """BernoulliArm - draws a reward of 1 w.p p
        """
    def __init__(self, probability):
        """Init

        Arguments:
            probability {[float]} -- [arm probability]
        """
        self.probability = probability

    def draw(self):
        """Draw reward based on instance probability """
        return scipy.stats.bernoulli.rvs(self.probability)

    def __repr__(self):
        return "BernoulliArm - {}".format(self.probability)
