import numpy as np

class WindyGridWorld:

    def __init__(self, nb_actions, gamma=1, stochastic=False) -> None:
        self.grid = np.zeros(shape=(7,10))
        self.starting_point = (3,0)
        self.goal = (3,7)
        self.wind = np.array([0,0,0,1,1,1,2,2,1,0])
        self.gamma = gamma
        self.stochastic = stochastic
        self.nb_actions = nb_actions
        if nb_actions == 4:
            self.legal_moves = {"S":(1,0), "N":(-1,0), "E":(0,1), "W":(0,-1)}
        elif nb_actions == 8:
            self.legal_moves = {
                "NW":(-1, -1), "N":(-1, 0), "NE":(-1, 1), "W":(0, -1), "E":(0, 1), "SW":(1, -1), "S":(1, 0), "SE":(1, 1),
            }
        elif nb_actions == 9:
            self.legal_moves = {
                "NW":(-1, -1), "N":(-1, 0), "NE":(-1, 1), "W":(0, -1), "E":(0, 1), "SW":(1, -1), "S":(1, 0), "SE":(1, 1), "still":(0,0),
            }

        self.pos = self.starting_point
        self.t = 0
        self.G = 0

    def step(self, move):
        movexy = self.legal_moves.get(move)
        #store wind effect
        wind = self.wind[self.pos[1]]
        if self.stochastic:
            wind += np.random.randint(-1,2)
        #move
        new_pos = (self.pos[0]+movexy[0], self.pos[1]+movexy[1])
        if not(self.is_oob(new_pos)):
            self.pos = new_pos
        #wind
        new_pos = (self.pos[0]-wind, self.pos[1])
        if not(self.is_oob(new_pos)):
            self.pos = new_pos
        # stats
        self.t +=1
        reward = -1
        self.G += reward
        return reward
        
    def is_terminal(self):
        return self.pos == self.goal

    def is_oob(self, pos):
        xmin, xmax, ymin, ymax = 0, self.grid.shape[1]-1, 0, self.grid.shape[0]-1
        y,x = pos
        if x>xmax or x<xmin or y<ymin or y>ymax:
            return True
        return False
    
    def reset_episode(self):
        self.__init__(nb_actions=self.nb_actions, gamma=self.gamma)

    def show(self):
        grid = self.grid.copy()
        grid[self.pos] = 1
        grid[self.goal] = 9
        print(grid)
        print("wind \n", np.array(self.wind, dtype=np.float32))
    


if __name__ == "__main__":

    env = WindyGridWorld(nb_actions=8)
    env.show()
    print(env.legal_moves)
    for i in range(15):
        print("timestep %d"%env.t)
        env.step("E")
        env.show()
    for i in range(4):
        print("timestep %d"%env.t)
        env.step("S")
        env.show()
    print("timestep %d"%env.t)
    env.step("W")
    env.show()
    print("timestep %d"%env.t)
    env.step("W")
    env.show()
    

