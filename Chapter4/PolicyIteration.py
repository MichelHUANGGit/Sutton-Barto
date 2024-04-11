import numpy as np
from CarRental import CarRental

def policy_evaluation(env, pi, v, epsilon):
    delta = 1000
    while delta > epsilon:
        delta = 0
        for s1 in range(len(v)):
            print("updating loc1 value:",s1)
            for s2 in range(len(v)):
                a = pi[s1,s2]
                v_temp = v[s1,s2]
                state = {"loc1" : s1, "loc2" : s2}
                # p(s1',s2'|s1,s2,a)
                p = env.transition_function(state,a)
                # r(s1,s2,a)
                r = env.reward_function(state,a)
                v[s1,s2] = np.sum(np.multiply(p, (r + env.gamma * v)))
                delta = max(delta, np.abs(v_temp - v[s1,s2]))
            print(delta)
    return v

def policy_improvement(env, pi, v):
    for s1 in range(len(v)):
        for s2 in range(len(v)):
            state = {"loc1" : s1, "loc2" : s2}
            a_temp = pi[s1,s2]
            q = np.zeros(len(pi))
            for a in range(len(q)):
                q[a] = np.sum(np.multiply(
                    env.transition_function(state,a), 
                    (env.reward_function(state,a) + env.gamma * v)
                ))
            pi[s1,s2] = np.argmax(q)
            is_policy_stable = (a_temp == pi[s1,s2])
    return pi, is_policy_stable

def policy_iteration(env, epsilon):
    v = np.zeros(shape=(env.max_cars+1, env.max_cars+1))
    pi = np.random.randint(-5,6, size=(env.max_cars+1, env.max_cars+1))

    is_policy_stable = False
    iteration = 0
    while not(is_policy_stable):
        iteration += 1
        print("Iteration:", iteration)
        v = policy_evaluation(env, pi, v, epsilon)
        pi, is_policy_stable = policy_improvement(env, pi, v)

    return v,pi

if __name__ == "__main__":
    jackscar = CarRental()
    v, pi = policy_iteration(jackscar, 1e-2)