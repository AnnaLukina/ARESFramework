# (c) Anna Lukina 28/05/2018
# "improved" particle swarm optimization (adaptive number of particles and steps ahead)
# ref: Amit Rathi "Optimization of Particle Swarm Optimization Algorithm"

import numpy as np
from Model import*


class PSO:
    gbest = 0 # global best solution
    pbest = 0 # personal best solution
    error = 0 # residual distance to the global optimum
    SwarmSize = 10
    MinNeighFrac = 0.25 # minimum fraction of
    MinNeighbors = MinNeighFrac * SwarmSize # minimum neighborhood size
    dim = 3 # problem dimensionality
    Num = 5 # number of agents
    c1 = 2 # cognitive coefficient
    c2 = 2 # social coefficient
    w_max = 1 # max inertia weight value
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

    def spawn(self, lb, ub, k, n):
        particle = np.random.uniform(lb,ub,[1,self.dim])
        if k > n:
            return particle
        else:
            k = k + 1
            return np.append(particle, PSO.spawn(self,lb,ub,k,n), axis=0)

    def step(self, model, horizon, lb, ub):
        best_fit = float("inf")
        # spawn positions and velocities for each particle in the swarm
        # each particle corresponds to the whole team of agents
        x = PSO.spawn(self,lb,ub,0, self.Num * self.SwarmSize)
        print("x:", x)
        v = PSO.spawn (self, lb-ub, ub-lb, 0, self.Num * self.SwarmSize)
        print ("v:", v)
        self.gbest = x
        self.pbest = x
        fit = model.J
        # compute fitness of each particle
        for p in range(1, self.SwarmSize):
            # proceed to the next state of the model using generated candidate action
            Model.move (model, x[(p-1) * self.Num : p * self.Num, 0 : self.dim], horizon)
            # compute fitness in this state
            fit = np.append(fit, model.J)
        print(fit)

        # main loop of PSO iterating for maximum number of iterations K_max
        self.K_i = self.K_min
        while self.K_i <= self.K_max:





