import numpy as np
from Racetrack import Racetrack



def OffPolicyMCControl(env, episodes):
    actions_id = np.arange(env.action_size)
    # ========= Utils ===========
    def simulate_trajectory(b):
        S,A,R = [], [], []
        while(not(env.terminal())):
            s = env.pos_vel_to_state_id[(env.pos,env.v)]
            S.append(s)
            a = np.random.choice(actions_id, p=b[s,:])
            A.append(a)
            r = env.step(env.action_id_to_tuple[a])
            R.append(r)
        return S,A,R

    def get_greedy_action(pi, s):
        # pi(s) = argmax_a Q(s,a)
        pi[s] = np.argmax(Q[s,:])

    # ======== Initialization ===========
    Q = np.random.randn(env.state_size, env.action_size)
    C = np.zeros(shape=(env.state_size, env.action_size))
    pi = np.zeros(shape=(env.state_size))
    for s in range(env.state_size):
        get_greedy_action(pi, s)

    # ========= Episode Loop ===========
    for episode in range(1, episodes+1):
        env.reset(new_episode=True)
        # b is an exploratory policy
        b = np.ones(shape=(env.state_size, env.action_size)) / env.action_size
        # Generate a trajectory following policy b
        S,A,R = simulate_trajectory(b)
        G = 0
        W = 1
        # =========== Trajectory Reverse Loop ===========
        for t in range(len(S)-1, -1, -1):
            G = env.gamma*G + R[t]
            C[S[t],A[t]] += W
            Q[S[t],A[t]] += W/C[S[t],A[t]] * (G - Q[S[t],A[t]])
            get_greedy_action(pi, S[t])
            if pi[S[t]] != A[t]:
                W *= 1/b[S[t],A[t]]
    return pi

if __name__ == "__main__":

    grid = np.array([
        [1,1,1,1,1,1,1,1,1,1],
        [1,0,1,1,1,1,1,1,1,1],
        [1,1,0,1,1,1,1,1,1,1],
        [1,1,1,0,1,1,1,1,1,1],
        [1,1,0,0,0,1,1,1,1,1],
        [1,1,0,1,0,0,1,1,1,1],
        [1,1,0,1,0,0,0,1,1,1],
        [1,1,0,1,0,0,0,0,1,1],
        [1,1,0,1,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1],
    ])
    starting_line = [(1,1)]
    finish_line = [(8,8)]
    max_velocity = 5
    p_cancel = 0

    env = Racetrack(grid, starting_line, finish_line, max_velocity, p_cancel)
    pi = OffPolicyMCControl(env, 1)
    print(pi)
    
