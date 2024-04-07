import numpy as np


class Bandit:

    def __init__(self, n, baseline_mean=0., noise_std=1.) -> None:
        self.n = n
        self.baseline_mean = baseline_mean
        self.noise_std = noise_std
        # The actual mean reward
        self.q = np.random.randn(n) + baseline_mean
        # timesteps
        self.t = 0
        self.cumulated_rewards = 0
        self.optimal_action = np.argmax(self.q)

    def step(self, a):
        # return the reward := mean reward + gaussian noise
        reward = self.q[a] + self.noise_std*np.random.randn()
        self.t += 1
        self.cumulated_rewards += reward
        return reward

    def reset(self):
        self.__init__(self.n)