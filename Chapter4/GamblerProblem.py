import numpy as np

class GamblerProblem:

    def __init__(self, goal, p_head) -> None:
        self.goal = goal
        self.p_head = p_head
        self.S = np.arange(goal+1)
        # self.s = np.random.randint(0, goal)
        self.s = goal//2

    def bet(self, a):
        return self.step(a)

    def step(self, a):
        is_head = (np.random.random()<=self.p_head)
        if is_head :
            self.s += a
        else :
            self.s -= a
    
    def is_terminal(self, s):
        return (s == 0) or (s==self.goal)
    
    def terminal_states(self):
        result = np.zeros(range(self.goal+1), dtype=bool)
        result[-1], result[0] = True, True
        return result

    def legal_bet(self, s=None):
        s = self.s if s is None else s
        return np.arange(0, min(s, 100-s)+1)

    # def transition_function(self,s,a):
    #     '''p(s',r|s,a)'''
    #     if a not in self.legal_bet(s) :
    #         print("illegal bet")
    #         exit()

    #     transition = np.zeros(shape=(self.goal+1, 2))
    #     if self.is_terminal(s+a) or self.is_terminal(s-a):
    #         transition[s+a,1] = self.p_head
    #         transition[s-a,0] = 1 - self.p_head
    #     else :
    #         transition[s+a,0] = self.p_head
    #         transition[s-a,0] = 1 - self.p_head
    #     return transition
    
    def transition_function(self):
        '''p(s',r|s,a) for all (s,a)'''

        transition = np.zeros(shape=(self.goal+1, 2, self.goal+1, len(self.S)))
        for s in self.S:
            # If terminal, then the transition probabilities are 0, don't do anything
            if self.is_terminal(s):
                continue
            # Else fill the transition matrix
            for a in self.legal_bet(s):
                if a == 0 :
                    continue
                if self.is_terminal(s+a):
                    transition[s+a,1,s,a] = self.p_head
                    transition[s-a,0,s,a] = 1 - self.p_head
                else :
                    transition[s+a,0,s,a] = self.p_head
                    transition[s-a,0,s,a] = 1 - self.p_head

        return transition
    
    def reward_function(self,s,a):
        NotImplemented

    
if __name__ == "__main__":
    env = GamblerProblem(100, 0.4)
    # print(env.s)
    # env.bet(50)
    # print(env.s)
    # print(env.is_terminal(env.s))
    print(env.legal_bet(50))
    # transition = env.transition_function(90, 10)
    # print(transition.sum())
    # print(transition[80:,:])