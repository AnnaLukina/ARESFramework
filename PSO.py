# (c) Anna Lukina 28/05/2018
# "improved" particle swarm optimization (adaptive number of particles and steps ahead)
# ref: Amit Rathi "Optimization of Particle Swarm Optimization Algorithm"

import numpy as np
from Model import*
#from joblib import Parallel, delayed
#import multiprocessing



class PSO:
    gbest = 0 # global best solution
    pbest = 0 # personal best solution
    best_fit = float("inf") # best fitness value so far
    error = 0 # residual distance to the global optimum
    SwarmSize = 10
    MinNeighFrac = 0.25 # minimum fraction of
    MinNeighbors = MinNeighFrac * SwarmSize # minimum neighborhood size
    dim = 3 # problem dimensionality
    Num = 5 # number of agents
    c1 = 2 # cognitive coefficient
    c2 = 2 # social coefficient
    w_max = 1.0 # max inertia weight value
    w_min = 0.1 # min inertia weight value
    K_i = 1 # total number of iterations
    K_min = 1 # first iteration
    K_max = 10 # last iteration

    def __init__(self, dim, num, swarm):
        self.dim = dim
        self.Num = num
        self.SwarmSize = swarm
        self.minNeighbors = max(2, round(self.SwarmSize * self.MinNeighFrac))
        self.K_max = max(10, self.Num * self.dim * 10)

    # generates a swarm of n uniformly distributed particles between lb and ub
    def spawn(self, lb, ub, k, n):
        particle = np.random.uniform(lb,ub,[1,self.dim])
        if k > n:
            return particle
        else:
            k = k + 1
            return np.append(particle, PSO.spawn(self,lb,ub,k,n), axis=0)

    # updates each particle p's position x and velocity v in the swarm
    # (improved PSO)
    def update(self, lb, ub, x, v):
        for p in range (1, self.SwarmSize):
            for d in range(1,self.dim):
                f1 = (1 - (self.K_i - self.K_max) / self.K_max)
                f2 = (self.K_i - self.K_min) / self.K_max
                v[p,d] = self.w_max * v[p,d] + f1 * np.random.uniform(0,1,1) * (self.pbest[p,d] - x[p,d]) + f2 * np.random.uniform(0,1,1) * (self.gbest[d] - x[p,d])
                x[p,d] = min(lb[d], max(ub[d], x[p,d] + v[p,d]))
        return x

    def step(self, model, horizon, lb, ub):
        # spawn positions and velocities for each particle in the swarm
        # each particle corresponds to the whole team of agents
        x = PSO.spawn(self,lb,ub,0, self.Num * self.SwarmSize)
        print("x:", x)
        v = PSO.spawn (self, lb-ub, ub-lb, 0, self.Num * self.SwarmSize)
        print ("v:", v)
        self.pbest = x
        fit = model.J
        # compute fitness of each particle
        for p in range(1, self.SwarmSize - 1):
            # proceed to the next state of the model using generated candidate action
            Model.move (model, x[p * self.Num : (p+1) * self.Num, 0 : self.dim], horizon)
            # compute fitness in this state
            fit = np.append(fit, model.J)
        print(fit)

        # main loop of PSO iterating for maximum number of iterations K_max
        self.K_i = self.K_min
        while self.K_i <= self.K_max:
            # find the global best fitness value and solution
            for k in range(1, self.SwarmSize):
                if fit[k] < self.best_fit:
                    self.best_fit = fit[k]
                    self.gbest = x[k * self.Num : (k+1) * self.Num, 0 : self.dim]
            # update particle velocities and positions
            x = PSO.update (self, lb, ub, x, v)
            # compute new fitness after update and compare with the previous one
            new_fit = model.J
            for p in range(1, self.SwarmSize - 1):
                Model.move (model, x[p * self.Num: (p + 1) * self.Num, 0: self.dim], horizon)
                new_fit = np.append(new_fit, model.J)
                if new_fit[p + 1] < fit[p + 1]:
                    fit[p + 1] = new_fit[p + 1]
            print(fit)
            self.K_i = self.K_i + 1







