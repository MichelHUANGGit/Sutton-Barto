import numpy as np
from WindyGridWorld import WindyGridWorld


class SARSA:

    def __init__(self, env):
        self.env = env
        self.Q = np.random.randn(*env.grid.shape, env.nb_actions)
        print("Q shape:",self.Q.shape)
        self.Q[*env.goal, :] = 0.
        self.idx_to_dir = {idx:direction for idx,direction in enumerate(env.legal_moves)}
        print("move idx to direction :",self.idx_to_dir)

    def epsilon_greedy_action(self, S, epsilon):
        if np.random.random()<epsilon:
            A = np.random.randint(low=0, high=self.env.nb_actions)
        else:
            A = np.argmax(self.Q[*S,:])
        return A

    def run_episode(self, step_size, epsilon):
        self.env.reset_episode()
        S = self.env.pos
        A = self.epsilon_greedy_action(S, epsilon)
        while not(self.env.is_terminal()):
            R = self.env.step(self.idx_to_dir.get(A))
            S_ = self.env.pos
            A_ = self.epsilon_greedy_action(S_, epsilon)
            self.Q[*S,A] += step_size * (R + self.env.gamma * self.Q[*S_,A_] - self.Q[*S,A])
            self.Q[*self.env.goal, :] = 0.
            S,A = S_, A_
        
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    nb_actions = [4,8,9,8]
    stochaticity = [False, False, False, True]
    n_envs = len(nb_actions)

    envs = [WindyGridWorld(nb_actions=nb_actions[i], gamma=1., stochastic=stochaticity[i]) for i in range(n_envs)]
    algorithms = [SARSA(envs[i]) for i in range(n_envs)]
    n_episodes = 170
    step_size = 0.5
    epsilon = 0.1

    timesteps_per_episodes = np.zeros(shape=(n_envs,n_episodes))
    for i in range(n_envs):
        for episode in range(1, n_episodes+1):
            # print("Episode %d"%episode)
            algorithms[i].run_episode(step_size=step_size, epsilon=epsilon)
            timesteps_per_episodes[i, episode-1] = envs[i].t


    cumulated_timesteps = np.cumsum(timesteps_per_episodes, axis=1)
    fig, ax = plt.subplots()
    for i in range(n_envs):
        ax.plot(
            cumulated_timesteps[i], np.arange(1, n_episodes+1), 
            label="nb_actions=%d stochastic %r"%(envs[i].nb_actions, stochaticity[i])
        )
    ax.legend()
    plt.show()