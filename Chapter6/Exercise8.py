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
    env1 = WindyGridWorld(nb_actions=8, gamma=1)
    print("LEGAL MOVES :",env1.legal_moves)
    algo1 = SARSA(env1)
    n_episodes = 170
    step_size = 0.5
    epsilon = 0.1

    timesteps_per_episodes = np.zeros(shape=(n_episodes,))
    for episode in range(1, n_episodes+1):
        print("Episode %d"%episode)
        algo1.run_episode(step_size=step_size, epsilon=epsilon)
        timesteps_per_episodes[episode-1] = env1.t

    cumulated_timesteps = np.cumsum(timesteps_per_episodes)
    plt.plot(cumulated_timesteps, np.arange(1, n_episodes+1))
    plt.axis(xmin=0, xmax=np.max(cumulated_timesteps))
    plt.show()