import numpy as np
import math
import jax

from numpy.random import default_rng
from gfn_maxent_rl.envs.dag_gfn.jraph_utils import to_graphs_tuple


class ReplayBuffer:
    def __init__(self, capacity, num_variables):
        self.capacity = capacity
        self.num_variables = num_variables

        nbytes = math.ceil((num_variables ** 2) / 8)
        dtype = np.dtype([
            # State (adjacency & mask)
            ('adjacency', np.uint8, (nbytes,)),
            ('mask', np.uint8, (nbytes,)),
            # Action
            ('action', np.int_, (1,)),
            # Reward
            ('reward', np.float_, (1,)),
            # Next state (adjacency & mask)
            ('next_adjacency', np.uint8, (nbytes,)),
            ('next_mask', np.uint8, (nbytes,))
        ])
        self._replay = np.zeros((capacity,), dtype=dtype)
        self._index = 0
        self._is_full = False

    def add(self, observations, actions, rewards, dones, next_observations):
        if np.all(dones):
            return True

        num_samples = np.sum(~dones)
        add_idx = np.arange(self._index, self._index + num_samples) % self.capacity
        self._is_full |= (self._index + num_samples >= self.capacity)
        self._index = (self._index + num_samples) % self.capacity

        data = {
            'adjacency': self.encode(observations['adjacency'][~dones]),
            'mask': self.encode(observations['mask'][~dones]),
            'action': actions[~dones],
            'reward': rewards[~dones],
            'next_adjacency': self.encode(next_observations['adjacency'][~dones]),
            'next_mask': self.encode(next_observations['mask'][~dones])
        }

        for name in data:
            shape = self._replay.dtype[name].shape
            self._replay[name][add_idx] = np.asarray(data[name].reshape(-1, *shape))

        return True

    def sample(self, batch_size, rng=default_rng()):
        indices = rng.choice(len(self), size=batch_size, replace=False)
        samples = self._replay[indices]

        adjacency = self.decode(samples['adjacency'])
        next_adjacency = self.decode(samples['next_adjacency'])

        total_num_edges = int(np.sum(adjacency))
        size = _nearest_power_of_2(total_num_edges)
        next_size = _nearest_power_of_2(total_num_edges + batch_size)

        return {
            'adjacency': adjacency,
            'graph': to_graphs_tuple(adjacency, size),
            'mask': self.decode(samples['mask']),
            'action': samples['action'],
            'reward': samples['reward'],
            'next_adjacency': next_adjacency,
            'next_graph': to_graphs_tuple(next_adjacency, next_size),
            'next_mask': self.decode(samples['next_mask']),
        }

    @property
    def dummy_samples(self):
        shape = (1, self.num_variables, self.num_variables)
        adjacency = np.zeros(shape, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.float32)
        graph = to_graphs_tuple(adjacency, 1)

        return {
            'adjacency': adjacency,
            'graph': graph,
            'mask': mask,
            'action': np.array([[self.num_variables ** 2]]),
            'reward': np.zeros((1, 1), dtype=np.float_),
            'next_adjacency': adjacency,
            'next_graph': graph,
            'next_mask': mask,
        }

    def __len__(self):
        return self.capacity if self._is_full else self._index

    def encode(self, decoded):
        encoded = decoded.reshape(-1, self.num_variables ** 2)
        return np.packbits(encoded.astype(np.int32), axis=1)

    def decode(self, encoded, dtype=np.float32):
        decoded = np.unpackbits(encoded, axis=-1, count=self.num_variables ** 2)
        decoded = decoded.reshape(*encoded.shape[:-1], self.num_variables, self.num_variables)
        return decoded.astype(dtype)


def _nearest_power_of_2(x):
    # https://stackoverflow.com/a/14267557
    return 1 if (x == 0) else (1 << (x - 1).bit_length())
