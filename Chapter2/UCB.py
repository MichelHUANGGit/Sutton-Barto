import numpy as np
from run_experiment import run_experiment
from Bandit import Bandit
from EpsilonGreedy import EpsilonGreedy
import os


class UCB:
    def __init__(self, c, timesteps) -> None:
        self.c = c
        self.timesteps = timesteps
        self.name = f"UCB c={c}"

    def run(self, env):
        self.N = np.zeros(env.n)
        self.R = np.zeros(env.n)
        self.Q = np.zeros(env.n)
        self.actions_taken = np.zeros(self.timesteps)
        self.average_reward = np.zeros(self.timesteps)
        self.optimal_action_taken = np.zeros(self.timesteps)

        for t in range(self.timesteps):
            # Play every action once before using UCB
            if t < env.n :
                a = t
            # UCB
            else :
                a = np.argmax(self.Q[:] + self.c * np.sqrt(np.log(t)/self.N[:]))

            self.actions_taken[t] = a
            self.N[a] += 1
            self.R[a] += env.step(a)
            self.Q[a] = self.R[a]/self.N[a]
            self.average_reward[t] = env.cumulated_rewards / (t+1)
            self.optimal_action_taken[t] = (self.actions_taken[:t+1] == env.optimal_action).mean()

        self.summary = {
            "average_reward" : self.average_reward,
            "pct_optimal_action" : self.optimal_action_taken,
        }

if __name__ == '__main__':
    N_BANDITS = 50
    N_ARMS = 10
    TIMESTEPS = 1000

    strats = [
        UCB(2, TIMESTEPS),
        EpsilonGreedy(0.10, TIMESTEPS),
    ]
    if not(os.path.exists("figures/")):
        os.mkdir("figures/")
    colors = ['blue', 'grey']

    env = Bandit(N_ARMS)
    run_experiment(env, strats, N_BANDITS, TIMESTEPS, colors, 'figures/Figure_2.3.png', figsize=(8,6))