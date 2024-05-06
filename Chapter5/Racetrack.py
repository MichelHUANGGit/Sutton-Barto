import numpy as np
import random

class Racetrack:

    INTERPOLATION_TICK = 100
    def __init__(self, grid, starting_line, finish_line, max_velocity, p_cancel, gamma=1) -> None:
        # 2D array, containing zeros, everywhere inside the track. 1 everywhere else. 
        # The indice (0,0) corresponds to the upper left cell.
        self.grid = grid
        # array of tuple (each tuple is an (y,x) coordinate)
        self.starting_line = starting_line
        self.finish_line = finish_line
        self.max_velocity = max_velocity
        self.p_cancel = p_cancel
        self.gamma = gamma
        self.reset()

        # Miscellaneous data
        self.pos_space = [(y,x) for y in range(0,grid.shape[0]) for x in range(0,grid.shape[1]) if grid[(y,x)]!=1]
        self.pos_size = len(self.pos_space)
        self.vel_space = [(vy,vx) for vy in range(-max_velocity,max_velocity+1,1) for vx in range(-max_velocity,max_velocity+1,1)]
        self.vel_size = len(self.vel_space)
        self.action_space = self.LegalActions()
        self.action_size = len(self.action_space)
        self.pos_vel_to_state_id = dict()
        self.state_id_to_pos_vel = dict()
        id = 0
        for pos in self.pos_space:
            for vel in self.vel_space:
                self.pos_vel_to_state_id[(pos,vel)] = id
                self.state_id_to_pos_vel[id] = (pos,vel)
                id += 1
        self.state_size = len(self.state_id_to_pos_vel)
        self.action_id_to_tuple = {id:tup for id, tup in enumerate(self.action_space)}
        self.tuple_to_action_id = {tup:id for id, tup in enumerate(self.action_space)}

    def LegalActions(self):
        return [(movey, movex) for movex in [-1,0,1] for movey in [-1,0,1]]

    def step(self, a:dict):
        # with probability p_cancel, the action is cancelled
        if np.random.random() > self.p_cancel:
            self.v = (self.v[0]+a[0], self.v[1]+a[1])
        self.t += 1
        self.G -= 1
        R = -1
        self.verify_path()
        return R
    
    def terminal(self):
        return self.pos in self.finish_line

    def get_projected_path(self):
        '''Computes all the intermediate positions (as if it were a continuous grid), then check if at any moment the car is out of bounds (OOB)
        The rule to determine if it's OOB is: 
        - if the majority the mass of the car is in one cell, then the car is considered to be in that cell. (we use the round() function)
        - if the car ever passes through a cell with value 1 (meaning it's OOB), then it'll be teleported back at the starting line
        Example :
        - Initial position (0,0), velocity = (1,1) in the direction (down, right), INTERPOLATION_TICK=10
        - the function will compute [(0.1,0.1), (0.2,0.2), ..., (0.4,0.4), (0.5,0.5), (0.6,0.6), ..., (1.0,1.0)]
        - then round the values ->  [(0  ,0  ), (0  ,0  ), ..., (0  ,0  ), (1  ,1  ), (1  ,1  ), ..., (1  ,1  )]
        - for the ties : round(0.5)=1
        '''
        interpolation = np.linspace(0,1,self.INTERPOLATION_TICK)
        return [(round(self.pos[0] + i*self.v[0]), round(self.pos[1] + i*self.v[1]))
                for i in interpolation]
    
    def verify_path(self):
        projected_path = self.get_projected_path()
        for pos in projected_path:
            if pos in self.finish_line:
                print("Terminal")
                self.pos = pos
                return None
            elif self.grid[pos]==1:
                print("OOB")
                self.reset(new_episode=False)
                return None
        self.pos = pos
    
    def reset(self, new_episode=True):
        # velocity tuple (vy,vx), positive values indicate for y (resp. x) downwards direction (resp. towards right)
        self.v = (0,0)
        # position (y,x)
        self.pos = random.choice(self.starting_line)
        if new_episode:
            # timestep
            self.t = 0
            # return
            self.G = 0

    def print(self):
        print("Timestep %d - vel (%d,%d) - pos (%d,%d)" %(self.t, *self.v, *self.pos))
        grid = self.grid.copy()
        grid[self.pos]=9
        print(grid)

if __name__ == "__main__":

    grid = np.array([
        [1,1,1,1,1],
        [1,0,1,1,1],
        [1,1,0,1,1],
        [1,0,0,0,1],
        [1,1,1,1,1],
    ])
    starting_line = [(1,1)]
    finish_line = [(3,3)]
    max_velocity = 5
    p_cancel = 0

    env = Racetrack(grid, starting_line, finish_line, max_velocity, p_cancel)
    env.print()
    actions = [(-1,1), (-1,1), (1,1)]
    for action in actions:
        env.step(action)
        env.print()
    # print(env.action_size)
    # print(env.pos_size, env.vel_size)
    # print(env.state_id_to_pos_vel)
    # print(env.state_size)