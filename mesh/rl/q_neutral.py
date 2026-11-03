"""Q-neutral: tabular Q-learning, one table shared by all agents."""
import pickle

import numpy as np

from mesh.rl.agents_io import (K_SLOTS, N_ACTIONS, action_mask, agent_features, alive_mask,
                               decode, greedy_with_sequential_channels, neighbor_slots)


def discretize(o, m):
    load_bin = int(np.digitize(o[0], [1.0, 5.0]))
    hop_bin = int(np.digitize(o[3], [1e-9, 0.1, 0.2]))  # gateway, near, mid, far (hop / n)
    radio1 = int(np.argmax(o[5:12]))
    slot = m.reshape(K_SLOTS, 4)
    rate = slot[:, 2] * 143.4 * slot[:, 0]
    best = int(np.argmax(rate)) if slot[:, 0].any() else -1
    rate_bin = int(np.digitize(rate.max(), [30.0, 80.0])) if best >= 0 else -1
    return (load_bin, bool(o[1] > 0.1), bool(o[2] > 0), hop_bin, radio1, best, rate_bin)


class QNeutral:
    def __init__(self, alpha=0.1, gamma=0.98, eps_start=0.6, eps_end=0.05,
                 eps_decay_steps=50_000, seed=0, seq_decode=True):
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        self.rng = np.random.default_rng(seed)
        self.q = {}
        self.steps = 0
        self.seq_decode = seq_decode
        self.last_action_idx = None

    def _eps(self):
        f = min(self.steps / self.eps_decay_steps, 1.0)
        return self.eps_start + f * (self.eps_end - self.eps_start)

    def _keys_mask(self, obs):
        slots = neighbor_slots(obs)
        o, m = agent_features(obs, slots)
        return [discretize(o[i], m[i]) for i in range(len(o))], action_mask(obs, slots), slots

    def _row(self, key):
        return self.q.setdefault(key, np.zeros(N_ACTIONS))

    def act(self, obs, explore=False):
        keys, mask, slots = self._keys_mask(obs)
        eps = self._eps() if explore else 0.0
        a = np.empty(len(keys), dtype=int)
        for i, k in enumerate(keys):
            valid = np.flatnonzero(mask[i])
            if self.rng.random() < eps:
                a[i] = self.rng.choice(valid)
            else:
                a[i] = valid[np.argmax(self._row(k)[valid])]
        if not explore and getattr(self, "seq_decode", True):
            joint = np.stack([np.where(mask[i], self._row(k), -np.inf) for i, k in enumerate(keys)])
            a = greedy_with_sequential_channels(obs, slots, joint)
        if explore:
            self.steps += 1
        self.last_action_idx = a
        return decode(obs, slots, a)

    def learn(self, obs, action_idx, rewards, next_obs, done):
        keys, _, _ = self._keys_mask(obs)
        next_keys, next_mask, _ = self._keys_mask(next_obs)
        alive = alive_mask(obs)
        for i in np.flatnonzero(alive):
            boot = 0.0 if done else self._row(next_keys[i])[next_mask[i]].max()
            row = self._row(keys[i])
            row[action_idx[i]] += self.alpha * (rewards[i] + self.gamma * boot - row[action_idx[i]])
        return {"table_size": len(self.q)}

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
