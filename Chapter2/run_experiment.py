from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


def run_experiment(env, strats, n_bandits, timesteps, colors, savefig=False, figsize=(8,6)):
    average_reward = np.zeros(shape=(n_bandits, timesteps))
    pct_optimal_action = np.zeros(shape=(n_bandits, timesteps))
    plot_data = dict()

    for strat in strats :
        for bandit in tqdm(range(n_bandits)):
            env.reset()
            strat.run(env)
            average_reward[bandit, :] = strat.average_reward
            pct_optimal_action[bandit, :] = strat.optimal_action_taken
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
