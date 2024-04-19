import numpy as np

def value_iteration(env, epsilon, verbose=False):

    # Initialization
    V = np.zeros(len(env.S))
    V[env.goal] = 1
    delta = epsilon + 1
    sweep = 0
    # Transition matrix p
    p = env.transition_function()
    # Bellman optimality equation
    while delta > epsilon:
        sweep += 1
        if verbose :
            print("Sweep :", sweep, "delta :", delta)
        delta = 0
        for s in env.S:
            # if s is terminal, do not change that value
            if env.is_terminal(s) :
                continue
            # otherwise apply the Bellman optimality equation
            old_value = V[s]
            V[s] = np.max([
                np.sum(p[:,0,s,a] * (0 + V[:]) + p[:,1,s,a] * (0 + V[:]))
                for a in env.legal_bet(s)
            ])
            delta = max(np.abs(V[s]-old_value), delta)
    print("Optimal value found")

    # Take the greedy policy wrt V*
    policy = np.zeros(len(env.S))
    for s in env.S :
        policy[s] = np.argmax([
            np.sum(p[:,0,s,a] * (0 + V[:]) + p[:,1,s,a] * (0 + V[:]))
            for a in env.legal_bet(s)
        ])
    policy[0], policy[env.goal] = 0, 0
    return V, policy

if __name__ == "__main__":
    from GamblerProblem import GamblerProblem
    env = GamblerProblem(100, 0.55)
    V, policy = value_iteration(env, 1e-8)
    print(dict(zip(range(len(V)), V)))

    from matplotlib import pyplot as plt

    plt.subplot(2,1,1)
    plt.plot(V[1:-1])
    plt.xticks([1, 25, 50, 75, 99])
    plt.subplot(2,1,2)
    plt.plot(policy[1:-1])
    plt.xticks([1, 25, 50, 75, 99])
    plt.show()
