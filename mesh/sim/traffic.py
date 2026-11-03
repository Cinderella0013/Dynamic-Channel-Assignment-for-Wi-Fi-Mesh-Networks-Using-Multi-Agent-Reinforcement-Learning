"""Per-node uplink demand: on/off Markov chain with exponential rate when on."""
import numpy as np

P_STAY = 0.9
P_START_ON = 0.7


class Traffic:
    def __init__(self, n, rng, mean_mbps, boost_nodes=None, boost_factor=3.0,
                 shift_step=None, shift_factor=3.0):
        self.rng = rng
        self.mean = mean_mbps
        self.on = rng.random(n) < P_START_ON
        self.scale = np.ones(n)
        if boost_nodes is not None:
            self.scale[np.asarray(boost_nodes, dtype=int)] *= boost_factor
        self.shift_step = shift_step
        self.shift_scale = np.ones(n)
        self.shift_scale[rng.permutation(n)[: n // 2]] = shift_factor

    def demand(self, t):
        """Stateful: call with consecutive t = 0, 1, 2, ..."""
        flip = self.rng.random(len(self.on)) >= P_STAY
        self.on = self.on ^ flip
        d = self.rng.exponential(self.mean, len(self.on)) * self.on * self.scale
        if self.shift_step is not None and t >= self.shift_step:
            d = d * self.shift_scale
        return d
