import numpy as np
from scipy.stats import poisson
from PoissonDiff import PoissonDiff

class CarRental :

    def __init__(
        self,
        price=10,
        move_car_cost=2,
        max_car_move=5,
        max_cars=20,
        gamma=0.9,
        poisson_lbda={
            "rental1" : 3,
            "rental2" : 4,
            "return1" : 3,
            "return2" : 2
        },
    ) -> None:
        self.poisson_lbda = poisson_lbda
        self.price = price
        self.move_car_cost = move_car_cost
        self.max_car_move = max_car_move
        self.max_cars = max_cars
        self.gamma = gamma

        self.state = {"loc1" : self.max_cars, "loc2" : self.max_cars}
        self.day = 0
        self.cumul_reward = 0
        self.discount = 1
        self.wait_return = {"loc1" : 0, "loc2" : 0}
        self.P1_diff = PoissonDiff(poisson_lbda["return1"], poisson_lbda["rental1"])
        self.P2_diff = PoissonDiff(poisson_lbda["return2"], poisson_lbda["rental2"])

    def legal_move(self):
        # We can move at most 5 cars (self.max_car_move) from one location to another, 
        # but there's also no point moving 5 cars from loc1 to loc2, if loc2 already has 18 cars since the max in 20, we would lose 3 cars
        # the result is a list ranging from negative numbers to positive numbers indicating how many cars we're moving from loc1 to loc2
        # if it's negative, it means we're moving from loc2 to loc1.
        return np.arange(
            start=-min(self.max_car_move, self.max_cars-self.state["loc1"]),
            stop=min(self.max_car_move, self.max_cars-self.state["loc2"])+1
        )
    
    def demand(self):
        '''
        Simulates the demand during the day
        '''
        self.day += 1
        # Simulate the demand, Poisson distribution
        events = {event : poisson.rvs(lbda) for event, lbda in self.poisson_lbda.items()}
        # can't rent more than what the location has
        events["rental1"] = min(events["rental1"], self.state["loc1"])
        events["rental2"] = min(events["rental2"], self.state["loc2"])
        reward = (events["rental1"] + events["rental2"]) * self.price * self.discount

        # Update the new_state <- old_state - rented cars + returned cars from yesterday (stocked in self.wait_return)
        # then clipped between 0 and self.max_cars
        self.state["loc1"] = np.clip(self.state["loc1"] - events["rental1"] + self.wait_return["loc1"], 0, self.max_cars)
        self.state["loc2"] = np.clip(self.state["loc1"] - events["rental2"] + self.wait_return["loc2"], 0, self.max_cars)
        self.wait_return["loc1"] = events["return1"]
        self.wait_return["loc2"] = events["return2"]

        # update discount factor
        self.discount *= self.gamma

        return reward

    def move(self, n):
        '''
        Actions during the night
        '''
        if n in self.legal_move():
            reward = 0
            # updating state
            self.state["loc1"] -= n
            self.state["loc2"] += n
            # rewards
            reward += -n*self.move_car_cost*self.discount
            reward += self.demand()
            self.cumul_reward += reward
            return reward
        else :
            print("Illegal Move")
            return None
        
    def transition_function(self, state, action):
        # transition probabilities p(s'|s,a)
        # s' is encoded by 2 dimensions : the number of cars in loc1 and loc2, we could also use one dimension of self.max_cars^2
        # i'm using the indexes s1 and s2 to represent s'
        p = np.zeros(shape=(self.max_cars, self.max_cars))
        for s1 in range(self.max_cars):
            for s2 in range(self.max_cars):
                # (10,10) current + (3,2) waiting - (3, 4) demand -> (9, 9)
                p[s1,s2] = self.P1_diff.pmf(s1 - state["loc1"]) * self.P1_diff.pmf(s2 - state["loc2"])
        return p


    def reward_function(self, state, action):
        pass

    def reset(self):
        pass

    def terminal(self):
        return (self.state["loc1"] == 0) & (self.state["loc2"] == 0)

if __name__ == "__main__":
    JackCar = CarRental()
    JackCar.state = {"loc1" : 10, "loc2" : 17}

    # for step in range(50):
    #     print("Day : %d - cumulated rewards : %d - loc1 : %d - loc2 : %d - wait1 : %d - wait2 : %d - discount : %.2f" 
    #         %(JackCar.day, JackCar.cumul_reward, JackCar.state["loc1"], JackCar.state["loc2"], JackCar.wait_return["return1"], JackCar.wait_return["return2"], JackCar.discount))
    #     action = np.random.choice(JackCar.legal_move())
    #     JackCar.move(action)
    p = JackCar.transition_function(JackCar.state, 0)
    print(p.shape)
    print(p.sum())
    print(p[:,19])