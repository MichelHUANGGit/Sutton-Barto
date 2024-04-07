import numpy as np
from run_experiment import run_experiment
import os
from Bandit import Bandit

class GradientBandit:
    
    def __init__(self, alpha, baseline, timesteps) -> None:
        self.timesteps = timesteps
        self.alpha = alpha
        self.baseline = baseline
        self.name = "Gradient " + r"$\alpha=$" + str(alpha) + " baseline=" + str(baseline)

    def softmax(self, a):
        exp = np.exp(a)
        return exp/sum(exp)
    
    def run(self, env):
        # Initialization
        self.H = np.zeros(env.n)
        self.R = np.zeros(self.timesteps)
        if not(self.baseline):
            baseline = 0.

        # Storing data for the plot
        self.actions_taken = np.zeros(self.timesteps)
        self.average_reward = np.zeros(self.timesteps)
        self.optimal_action_taken = np.zeros(self.timesteps)

        for t in range(self.timesteps):
            pi = self.softmax(self.H)
            A = np.random.choice(range(env.n), p=pi)
            self.actions_taken[t] = A
            self.R[t] = env.step(A)
            if self.baseline :
                # I can compute the baseline using the env class directly,
                # baseline = env.cumulated_rewards / (t+1)
                # but we also could do:
                baseline = np.mean(self.R[:t+1])
            self.H = self.H + self.alpha * (self.R[t] - baseline) * (np.equal(np.arange(env.n),A) - pi)
            self.average_reward[t] = env.cumulated_rewards / (t+1)
            self.optimal_action_taken[t] = (self.actions_taken[:t+1] == env.optimal_action).mean()

        self.summary = {
            "average_reward" : self.average_reward,
            "pct_optimal_action" : self.optimal_action_taken,
        }


if __name__ == '__main__':
    N_BANDITS = 2000
    N_ARMS = 10
    TIMESTEPS = 1000

    strats = [
        GradientBandit(0.1, True, TIMESTEPS),
        GradientBandit(0.4, True, TIMESTEPS),
        GradientBandit(0.1, False, TIMESTEPS),
        GradientBandit(0.4, False, TIMESTEPS),
    ]
    if not(os.path.exists("figures/")):
        os.mkdir("figures/")
    colors = ['blue', 'lightblue', 'brown', 'burlywood']

    env = Bandit(N_ARMS, baseline_mean=+4.)
    run_experiment(env, strats, N_BANDITS, TIMESTEPS, colors, savefig="figures/Figure_2.4.png", figsize=(8,6))