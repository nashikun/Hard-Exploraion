from collections import deque
import random
import numpy as np
import torch


class Memory:
    def __init__(self):
        self.buffer = []

    def get_batch(self) -> tuple:
        s_batch = torch.stack([sample[0] for sample in self.buffer])
        r_batch = torch.Tensor([sample[1] for sample in self.buffer])
        d_batch = torch.Tensor([sample[2] for sample in self.buffer]) + 0
        s2_batch = torch.stack([sample[3] for sample in self.buffer])
        log_prob_batch = torch.stack([sample[4] for sample in self.buffer])
        entropy_batch = torch.stack([sample[5] for sample in self.buffer])
        cumulative_rewards_batch = torch.stack([sample[6] for sample in self.buffer])
        # Reset the buffer
        self.buffer = []
        return s_batch, r_batch, d_batch, s2_batch, log_prob_batch, entropy_batch, cumulative_rewards_batch

    def extend(self, cumulative_rewards):
        n, m = len(self.buffer), len(cumulative_rewards)
        for i in range(m):
            self.buffer[n - 1 - i] = self.buffer[n - 1 - i] + (cumulative_rewards[m - 1 - i],)

    def store(self, s, r, d, s_p, log_prob, entropy) -> None:
        e = (s, r, d, s_p, log_prob, entropy)
        self.buffer.append(e)

    def __len__(self):
        return len(self.buffer)


class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def sample(self, batch_size: int, randomized: bool) -> tuple:
        if randomized:
            samples = random.sample(self.buffer, min(batch_size, len(self)))
        else:
            samples = self.buffer[-min(batch_size, len(self)):]

        s_batch = torch.stack([sample[0] for sample in samples])
        a_batch = torch.Tensor([sample[1] for sample in samples])
        r_batch = torch.Tensor([sample[2] for sample in samples])
        d_batch = torch.Tensor([sample[3] for sample in samples])
        s2_batch = torch.stack([sample[4] for sample in samples])
        return s_batch, a_batch, r_batch, d_batch, s2_batch

    def store(self, s, a, r, d, s_p) -> None:
        e = (s, a, r, d, s_p)
        self.buffer.append(e)

    def __len__(self) -> int:
        return len(self.buffer)
