import numpy as np
from scipy.stats import poisson
from PoissonDiff import TruncatedPoissonDiff
from tqdm import tqdm

class CarRental :
    '''
    Jack's Car Rental problem from the Reinforcement Learning Book (R. Sutton & A. Barto)
    There are a few ambiguities that aren't clarified in the book :
    - Do the returned cars take storage place in the locations when they're not available yet for rent ?
    - If I have 0 cars in location 1 at night, but I have 2 returned cars from today that will be available for tomorrow, 
    does that mean we can rent 2 cars, even though yesterday night, the state was 0 cars at location 1. 
    -> these are extra nuances to consider when modeling the reward function and transitions
    '''
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
        # This variable represents the 2 random variables, one for each location :
        # for loc1, it represents the random variable X-Y where X is the nb of returned cars -> Poisson(3), Y is the nb of rented cars -> Poisson(3)
        # for loc2, it represents the random variable X-Y where X is the nb of returned cars -> Poisson(2), Y is the nb of rented cars -> Poisson(4)
        self.plus_minus = TruncatedPoissonDiff(poisson_lbda, max_cars)
        # self.store_transition_tensor()
        # self.store_reward_tensor()

    def legal_move(self, s):
        # We can move at most 5 cars (self.max_car_move) from one location to another, 
        # but there's also no point moving 5 cars from loc1 to loc2, if loc2 already has 18 cars since the max in 20, we would lose 3 cars
        # the result is a list ranging from negative numbers to positive numbers indicating how many cars we're moving from loc1 to loc2
        # if it's negative, it means we're moving from loc2 to loc1.
        return np.arange(
            start=-min(self.max_car_move, self.max_cars-s["loc1"]),
            stop=min(self.max_car_move, self.max_cars-s["loc2"])+1
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

    def move(self, a):
        '''
        Actions during the night
        '''
        if a in self.legal_move(self.state):
            reward = 0
            # updating state
            self.state["loc1"] -= a
            self.state["loc2"] += a
            # rewards
            reward += -a*self.move_car_cost*self.discount
            reward += self.demand()
            self.cumul_reward += reward
            return reward
        else :
            print("Illegal Move")
            return None
        
    def transition_function(self, s, a):
        # p(s'|s,a), a 21x21 array, one dim for each location
        return self.plus_minus.transition_proba(s,a)


    def reward_function(self, s, a):
        # r(s,a) the average reward for taking action a in state s
        moved_car_cost = np.abs(a) * self.move_car_cost

        max_rent1 = s["loc1"]-a
        support = np.arange(0, max_rent1)
        loc1_reward = np.sum(poisson.pmf(support, self.poisson_lbda["rental1"]) * support * self.price)
        truncated_support = np.arange(max_rent1, 150)
        loc1_reward += np.sum(poisson.pmf(truncated_support, self.poisson_lbda["rental1"]) * max_rent1 * self.price)

        max_rent2 = s["loc2"]+a
        support = np.arange(0, max_rent2)
        loc2_reward = np.sum(poisson.pmf(support, self.poisson_lbda["rental2"]) * support * self.price)
        truncated_support = np.arange(max_rent2, 150)
        loc2_reward += np.sum(poisson.pmf(truncated_support, self.poisson_lbda["rental2"]) * max_rent2 * self.price)
        return loc1_reward + loc2_reward - moved_car_cost

    def store_transition_tensor(self):
        # stores p(s1',s2'|s1,s2,a) for all state-action triplets (s1,s2,a)
        # for 20 max cars, 5 cars moved at most, this becomes a tensor of shape (21,21,21,21,11), size ~ 2M floats
        d = self.max_cars+1
        moves = 2*self.max_car_move+1
        self.p = np.zeros(shape=(d,d,d,d,moves))
        print("Storing the transition tensor...")
        for s1 in tqdm(range(d)):
            print("Computing the transition probabilities starting from state s1=",s1)
            for s2 in range(d):
                for a in self.legal_move(s={"loc1":s1, "loc2":s2}):
                    self.p[:,:,s1,s2,a] = self.transition_function({"loc1":s1, "loc2":s2}, a)
        print("Done.")

    def store_reward_tensor(self):
        # stores r(s1,s2,a) for all state-action triplets (s1,s2,a)
        d = self.max_cars+1
        moves = 2*self.max_car_move+1
        self.r = np.zeros(shape=(d,d,moves))
        print("Storing the reward tensor...")
        for s1 in range(d):
            print("Computing the reward functions starting from state s1=",s1)
            for s2 in range(d):
                for a in self.legal_move(s={"loc1":s1, "loc2":s2}):
                    self.r[s1,s2,a] = self.reward_function({"loc1":s1, "loc2":s2}, a)
        print("Done.")
        

    def reset(self):
        pass

    def terminal(self):
        return (self.state["loc1"] == 0) & (self.state["loc2"] == 0)

if __name__ == "__main__":
    JackCar = CarRental()
    JackCar.state = {"loc1" : 0, "loc2" : 2}

    # for step in range(50):
    #     print("Day : %d - cumulated rewards : %d - loc1 : %d - loc2 : %d - wait1 : %d - wait2 : %d - discount : %.2f" 
    #         %(JackCar.day, JackCar.cumul_reward, JackCar.state["loc1"], JackCar.state["loc2"], JackCar.wait_return["return1"], JackCar.wait_return["return2"], JackCar.discount))
    #     action = np.random.choice(JackCar.legal_move())
    #     JackCar.move(action)
    # p = JackCar.transition_function(JackCar.state, 0)
    # print(p.shape)
    # print(p.sum())
    print(JackCar.p.shape)
