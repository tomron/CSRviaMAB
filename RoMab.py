"""R-O MAB"""

class RoMab():
    """R-O MAB"""
    def __init__(self, arms, T, f, lam):
        self.arms = arms
        self.T = T
        self.f = f
        self.lam = lam

    def __str__(self):
        arms_str = list(a.probability for a in self.arms)
        return (f"arms: {arms_str}, "
                f"T: {self.T}, f: {self.f}, "
                f"lambda: {self.lam}")

    def get_l(self):
        """Function lipschitz constant"""
        return self.f.get_l()
