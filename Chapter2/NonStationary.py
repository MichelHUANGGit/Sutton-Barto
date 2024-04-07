from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import os

class NonStarionaryBandit:

    def __init__(self, n, random_walk_std, noise_mean=0., noise_std=1.) -> None:
        self.n = n #number of arms
        self.random_walk_std = random_walk_std
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        # The actual mean reward start at 0, then each action action does its own random walk
        self.q = np.zeros(n)
        # timesteps
        self.t = 0
        self.cumulated_rewards = 0
        self.optimal_action = np.argmax(self.q)

    def step(self, a):
        # return the reward := mean reward + gaussian noise
        reward = self.q[a] + self.noise_mean + self.noise_std*np.random.randn()
        self.t += 1
        self.cumulated_rewards += reward
        # After the step, the real action value changes, all the q(a) take independent random walks
        self.q += self.random_walk_std * np.random.randn(self.n)
        self.optimal_action = np.argmax(self.q)
        return reward

    def reset(self):
        self.__init__(self.n, self.random_walk_std)

def randargmax(a):
    return np.random.choice(np.flatnonzero(a == a.max()))

class EpsilonGreedy:

    def __init__(self, epsilon, alpha, timesteps, name=None) -> None:
        self.epsilon = epsilon
        self.alpha = alpha
        self.timesteps = timesteps
        self.name = name if name is not None else r"$\epsilon = $" + str(epsilon) 

    def run(self, env):
        N = np.zeros(env.n)
        self.Q = np.zeros(shape=(env.n))
        # Plot data
        average_reward = np.zeros(self.timesteps)
        pct_optimal_action = np.zeros(self.timesteps)
        optimal_action_taken = 0

        for t in range(self.timesteps):
            # Epsilon greedy
            if np.random.random() < self.epsilon :
                A = np.random.randint(0, env.n)
            else :
                A = randargmax(self.Q)

            N[A] += 1
            R = env.step(A)
            step_size = self.alpha(N[A])
            self.Q[A] += step_size * (R-self.Q[A])

            # Plot data
            average_reward[t] = env.cumulated_rewards / (t+1)
            optimal_action_taken += int(env.optimal_action == A)
            pct_optimal_action[t] = optimal_action_taken / (t+1)

        self.summary = {
            "average_reward" : average_reward,
            "pct_optimal_action" : pct_optimal_action,
        }


def run_experiment(env, strats, n_bandits, timesteps, colors, savefig=False, figsize=(8,6)):
    average_reward = np.zeros(shape=(n_bandits, timesteps))
    pct_optimal_action = np.zeros(shape=(n_bandits, timesteps))
    plot_data = dict()

    for strat in strats :
        for bandit in tqdm(range(n_bandits)):
            env.reset()
            strat.run(env)
            average_reward[bandit, :] = strat.summary["average_reward"]
            pct_optimal_action[bandit, :] = strat.summary["pct_optimal_action"]
        # compute the average across all bandits
        plot_data[f'average_reward_{strat.name}'] = np.mean(average_reward, axis=0)
        plot_data[f'pct_optimal_action_{strat.name}'] = np.mean(pct_optimal_action, axis=0)

    plt.figure(figsize=figsize)
    plt.subplot(2,1,1)
    for i, strat in enumerate(strats):
        plt.plot(plot_data[f'average_reward_{strat.name}'], colors[i], label=strat.name)
        plt.ylabel("Average reward")
        plt.xlabel("Steps")
        plt.legend()
    plt.subplot(2,1,2)
    for i, strat in enumerate(strats):
        plt.plot(plot_data[f'pct_optimal_action_{strat.name}'], colors[i], label=strat.name)
        plt.ylabel("% Optimal action taken")
        plt.xlabel("Steps")
        plt.legend()
    if savefig :
        plt.savefig(savefig)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    N_ARMS = 10
    TIMESTEPS = 1500
    N_BANDITS = 1000
    RANDOM_WALK_STD = 0.5

    env = NonStarionaryBandit(
        n=N_ARMS,
        random_walk_std=RANDOM_WALK_STD
    )

    strats = [
        EpsilonGreedy(epsilon=0.1, alpha=(lambda k : 1/k), timesteps=TIMESTEPS, name=r"$\epsilon = 0.1$ $\alpha=\frac{1}{k}$"),
        EpsilonGreedy(epsilon=0.1, alpha=(lambda k : 0.1), timesteps=TIMESTEPS, name=r"$\epsilon = 0.1$ $\alpha=0.1$"),
    ]
    colors = ['green', 'red']
    if not(os.path.exists("figures/")):
        os.mkdir("figures/")
    run_experiment(env, strats, N_BANDITS, TIMESTEPS, colors, False)