import numpy as np
from Bandit import Bandit
from run_experiment import run_experiment

def randargmax(a):
    return np.random.choice(np.flatnonzero(a == a.max()))

class EpsilonGreedy:

    def __init__(self, epsilon, timesteps) -> None:
        self.epsilon = epsilon
        self.timesteps = timesteps
        self.name = r"$\epsilon = $" + str(epsilon)

    def run(self, env):
        self.N = np.zeros(env.n)
        self.R = np.zeros(env.n)
        self.Q = np.zeros(env.n)
        self.actions_taken = np.zeros(self.timesteps)
        self.average_reward = np.zeros(self.timesteps)
        self.optimal_action_taken = np.zeros(self.timesteps)

        for t in range(self.timesteps):
            # Epsilon greedy
            if np.random.random() < self.epsilon :
                a = np.random.randint(0, env.n)
            else :
                a = randargmax(self.Q)

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
    N_BANDITS = 2000
    N_ARMS = 10
    TIMESTEPS = 1000

    strat1 = EpsilonGreedy(0, TIMESTEPS)
    strat2 = EpsilonGreedy(0.01, TIMESTEPS)
    strat3 = EpsilonGreedy(0.10, TIMESTEPS)
    strats = [strat1, strat2, strat3]
    colors = ['green', 'red', 'black']

    env = Bandit(N_ARMS)
    run_experiment(env, strats, N_BANDITS, TIMESTEPS, colors, 'figures/Figure_2.1.png')