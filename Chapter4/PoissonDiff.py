import numpy as np
from scipy.stats import poisson


class PoissonDifference:
    '''
    Distribution of the random variable X - Y when X, Y both follow a Poisson distribution
    '''
    MAX = 150
    def __init__(self, lbdaX, lbdaY) -> None:
        self.lbdaX = lbdaX
        self.lbdaY = lbdaY
        self.pmfX = poisson.pmf(np.arange(self.MAX), lbdaX)
        self.pmfY = poisson.pmf(np.arange(self.MAX), lbdaY)

    def pmf(self, n):
        '''
        P(X-Y = n)
        '''
        if n==0:
            return np.dot(self.pmfX, self.pmfY)
        elif n<0:
            # return np.dot(self.pmfX[:-n], self.pmfY[n:])
            return np.sum([self.pmfX[k+n]*self.pmfY[k] for k in range(-n, self.MAX)])
        else :
            return np.sum([self.pmfX[k+n]*self.pmfY[k] for k in range(self.MAX-n)])
        
    def sf(self, n):
        # P(X-Y > n) = sum P(X-Y = k) for k>n
        return np.sum([self.pmf(k) for k in range(n+1,self.MAX)])


class TruncatedPoissonDiff:
    def __init__(self, lbdas, s, a, max_cars) -> None:
        self.lbdas = lbdas
        self.max_cars = max_cars
        self.s = s
        self.a = a

        self.plus_minus1 = PoissonDifference(lbdas["return1"], lbdas["demand1"])
        self.plus_minus2 = PoissonDifference(lbdas["return2"], lbdas["demand2"])
        '''
        Example :
        when location 1 has s1 = 15 cars, s2 = 15 cars, and we moved 3 cars from location 2 to 1 (a=-3)
        What is the probability of transitioning to state (s1',s2')=(20,15) ?
        Well we already know that 3 cars are transferred during the night, so we already have s1 = 18, s2 = 12 (intermediate state)
        Now there are several ways we could get to (s1',s2')=(20,15), for location 1:
        - Either no one rents a car, and 2 cars return, at the end of the day s1' = 20
        - or no one rents a car, and 3 cars return, we'd have 21 cars, but it is truncated to max_cars = 20, so s1' = 20
        - there are many other possibilities, we have to sum over all of these probabilities.
        Same logic for location 2.
        I define plus_minus1 as the number of returned cars minus the number of rented cars of location 1
        '''

    def transition_proba(self):
        # transition probabilities p(s'|s,a)
        # s' is encoded by 2 dimensions : the number of cars in loc1 and loc2, we could also use one dimension of self.max_cars^2
        # i'm using the indexes s1 and s2 to represent s'
        d = self.max_cars
        p = np.zeros(shape=(d+1, d+1))
        for s1 in range(1,d):
            for s2 in range(1,d):
                # Example : (10,10) current + plus_minus + (+2,-2) action -> (9, 9) final
                # So, probability(final) = probability(plus_minus = final - current - action)
                # Little nuance : action is negative when moving cars from loc2 to loc1
                p[s1,s2] = self.plus_minus1.pmf(s1 - self.s["loc1"] - self.a) * self.plus_minus2.pmf(s2 - self.s["loc2"] + self.a)

        # We then have to consider the truncating effect, by adding the remaining mass of the truncated values
        for s2 in range(d):
            # probability of s1 >= 20 (too many cars returned)
            p[d,s2] = self.plus_minus1.sf(d - self.s["loc1"] - self.a -1) * self.plus_minus2.pmf(s2 - self.s["loc2"] + self.a)
            # probability of s1 <=0 (too many rents for our stock)
            p[0,s2] = (1-self.plus_minus1.sf(0 - self.s["loc1"] - self.a)) * self.plus_minus2.pmf(s2 - self.s["loc2"] + self.a)
        for s1 in range(d):
            p[s1,d] = self.plus_minus1.pmf(s1 - self.s["loc1"] - self.a) * self.plus_minus2.sf(d - self.s["loc2"] + self.a -1)
            p[s1,0] = self.plus_minus1.pmf(s1 - self.s["loc1"] - self.a) * (1-self.plus_minus2.sf(0 - self.s["loc2"] + self.a))

        return p
    
        '''# X - Y
        # X is the return, and is truncated to max_return = max_cars-current_cars
        max_return = self.max_cars - state["loc2"]
        self.return_pmf2 = poisson.pmf(np.arange(max_return + 1), lbdas["return2"])
        self.return_pmf2[max_return] = poisson.sf(max_return-1, lbdas["return2"])
        # same thing for location 1
        max_return = self.max_cars - state["loc1"]
        self.return_pmf1 = poisson.pmf(np.arange(max_return + 1), lbdas["return1"])
        self.return_pmf1[max_return] = poisson.sf(max_return-1, lbdas["return1"])

        # Y is the demand, if greater than the current stock, then it will be truncated to current stock
        max_demand = state["loc2"]
        self.demand_pmf2 = poisson.pmf(np.arange(max_demand+1), lbdas["demand2"])
        self.demand_pmf2[max_demand] = poisson.sf(max_demand-1, lbdas["demand2"])
        max_demand = state["loc1"]
        self.demand_pmf1 = poisson.pmf(np.arange(max_demand+1), lbdas["demand1"])
        self.demand_pmf1[max_demand] = poisson.sf(max_demand-1, lbdas["demand1"])'''

if __name__ == "__main__":
    state = {"loc1" : 18, "loc2" : 10}
    lbdas = {
        "return1" :3,
        "return2" :2,
        "demand1" :3,
        "demand2" :4,
    }
    # Poisson = TruncatedPoissonDiff(lbdas, state, 0, 20)
    # print(Poisson.transition_proba().sum())

    PDIFF = PoissonDifference(2,4)
    print(PDIFF.pmf(-1))
    print(PDIFF.sf(0))