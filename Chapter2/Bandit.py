import numpy as np

class Bandit:

    def __init__(self, n) -> None:
        self.n = n
        # The actual mean reward
        self.q = np.random.randn(n)
        # timesteps
        self.t = 0
        self.cumulated_rewards = 0
        self.optimal_action = np.argmax(self.q)

    def step(self, a):
        # return the reward := mean reward + gaussian noise
        reward = self.q[a] + np.random.randn()
        self.t += 1
        self.cumulated_rewards += reward
        return reward

    def reset(self):
        self.__init__(self.n)