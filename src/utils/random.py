import random
import torch
import numpy as np
from numpy.random import MT19937, RandomState


class PersistentRandom:
    def __init__(self, seed=None):
        rs = RandomState(seed)
        mt19937 = MT19937()
        mt19937.state = rs.get_state()
        self.random_state = RandomState(mt19937)

    def seed(self, seed):
        self.random_state.seed(seed)

    def permutation(self, x):
        return self.random_state.permutation(x)

    def sample(self, population, k):
        return self.random_state.choice(population, size=k, replace=False)


def set_deterministic(seed):
    # fix random seeds for reproducibility
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False # comment for faster running
    np.random.seed(seed)
    random.seed(seed)
