"""Per-node uplink demand: on/off Markov chain with exponential rate when on."""
import numpy as np

P_STAY = 0.9
P_START_ON = 0.7


class Traffic:
    def __init__(self, n, rng, mean_mbps, boost_nodes=None, boost_factor=3.0,
                 shift_step=None, shift_factor=3.0, periodic=None, shift_period=None, static=False):
        self.rng = rng
        self.mean = mean_mbps
        self.on = rng.random(n) < P_START_ON
        self.scale = np.ones(n)
        if boost_nodes is not None:
            self.scale[np.asarray(boost_nodes, dtype=int)] *= boost_factor
        self.shift_step = shift_step
        self.shift_scale = np.ones(n)
        self.shift_scale[rng.permutation(n)[: n // 2]] = shift_factor
        self.boost_factor, self.shift_factor = boost_factor, shift_factor
        self.periodic, self.shift_period, self.static = periodic, shift_period, static
        # separate stream so the S1-S6 draw sequence is unchanged when shift_period is unused
        self.shift_rng = np.random.default_rng(rng.integers(2**31)) if shift_period else None

    def set_boost(self, nodes):
        self.scale = np.ones(len(self.on))
        self.scale[np.asarray(nodes, dtype=int)] *= self.boost_factor

    def demand(self, t):
        """Stateful: call with consecutive t = 0, 1, 2, ..."""
        if self.static:  # C2: every node always on at the mean rate
            return np.full(len(self.on), self.mean) * self.scale
        flip = self.rng.random(len(self.on)) >= P_STAY
        self.on = self.on ^ flip
        d = self.rng.exponential(self.mean, len(self.on)) * self.on * self.scale
        if self.shift_step is not None and t >= self.shift_step:
            d = d * self.shift_scale
        if self.shift_period and t % self.shift_period == 0:  # B5: re-draw the boosted half
            n = len(self.on)
            self.shift_scale = np.ones(n)
            self.shift_scale[self.shift_rng.permutation(n)[: n // 2]] = self.shift_factor
        if self.shift_period:
            d = d * self.shift_scale
        if self.periodic:  # B2
            d = d * (1 + 0.8 * np.sin(2 * np.pi * t / self.periodic))
        return d
