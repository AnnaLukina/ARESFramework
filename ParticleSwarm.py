# (c) Anna Lukina 28/05/2018
# "improved" particle swarm optimization (adaptive number of particles and steps ahead)
# ref: Amit Rathi "Optimization of Particle Swarm Optimization Algorithm"

import numpy as np
from DynModel import*
#from joblib import Parallel, delayed
#import multiprocessing



class ParticleSwarm:
    gbest = 0 # global best solution
    pbest = 0 # personal best solution
    gbest_fit = float("inf") # best global fitness value so far
    pbest_fit = 0 # best personal fitness value so far
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
    w = w_min # inertia
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
    def spawn(self, lb, ub, k):
        # an array of particles for each bird
        # size of the swarm is the number of such arrays
        particle = np.subtract(ub,lb) * np.random.random_sample((self.Num,self.dim)) + lb
        if k > self.SwarmSize:
            return particle
        else:
            k = k + 1
            return np.append(particle, ParticleSwarm.spawn(self,lb,ub,k), axis=0)

    # updates each particle p's position x and velocity v in the swarm
    # (improved PSO)
    def update(self, lb, ub, x, v):
        for p in range (0, self.SwarmSize - 1):
            f1 = (1 - (self.K_i - self.K_max) / self.K_max) * np.random.uniform(0,1,1)
            f2 = (self.K_i - self.K_min) / self.K_max * np.random.uniform(0,1,1)
            v[p * self.Num : (p+1) * self.Num, 0 : self.dim] = self.w * v[p * self.Num : (p+1) * self.Num, 0 : self.dim] + f1 * (self.pbest[p * self.Num : (p+1) * self.Num, 0 : self.dim] - x[p * self.Num : (p+1) * self.Num, 0 : self.dim]) + f2 * (self.gbest - x[p * self.Num : (p+1) * self.Num, 0 : self.dim])
            x[p * self.Num : (p+1) * self.Num, 0 : self.dim] = x[p * self.Num : (p+1) * self.Num, 0 : self.dim] + v[p * self.Num : (p+1) * self.Num, 0 : self.dim]
        # check constraints on control variables
        for p in range(0, self.SwarmSize - 1):
            for i in range(0,self.Num-1):
                for j in range(0,self.dim-1):
                    if x[p + i,j] < lb[i,j]:
                        x[p + i,j] = lb[i,j]
                    if x[p + i,j] > ub[i,j]:
                        x[p + i,j] = ub[i,j]
        return x

    def step(self, model, horizon, lb, ub):
        w_count = 0
        # spawn positions and velocities for each particle in the swarm
        # each particle corresponds to the whole team of agents
        x = ParticleSwarm.spawn(self,lb,ub,0)
        v = ParticleSwarm.spawn (self, np.subtract(lb,ub), np.subtract(ub,lb), 0)
        self.pbest = x
        fit = model.J
        # compute fitness of each particle
        for j in range(1, self.SwarmSize - 1):
            # proceed to the next state of the model using generated candidate action
            DynModel.move (model, x[j*self.Num:(j+1)*self.Num,0:self.dim], horizon)
            # compute fitness in this state
            fit = np.append(fit, model.J)
        self.pbest_fit = fit # initialize best personal fitness
        print("fit:",fit)

        # main loop of PSO iterating for maximum number of iterations K_max
        self.K_i = self.K_min
        while self.K_i <= self.K_max:
            # find the global best fitness value and solution
            for k in range(0, self.SwarmSize - 1):
                if fit[k] < self.gbest_fit:
                    self.gbest_fit = fit[k]
                    self.gbest = x[k * self.Num : (k+1) * self.Num, 0 : self.dim]

            # update particle velocities and positions
            x = ParticleSwarm.update (self, lb, ub, x, v)
            # compute new fitness after update and compare with the previous one
            new_fit = model.J
            for p in range(1, self.SwarmSize - 1):
                DynModel.move (model, x[p * self.Num: (p + 1) * self.Num, 0: self.dim], horizon)
                new_fit = np.append(new_fit, model.J)

            for p in range(0, self.SwarmSize-1):
                if new_fit[p] < self.pbest_fit[p]:
                    self.pbest_fit[p] = new_fit[p]
                    self.pbest[p] = x[p]
                if new_fit[p] < self.gbest_fit:
                    self.gbest_fit = new_fit[p]
                    self.gbest = x[p]
                    w_count = max(0, w_count - 1)
                else:
                    w_count = w_count + 1
            #print("update:", self.gbest_fit)
            self.K_i = self.K_i + 1

            # update inertia
            if (w_count < 2):
                self.w = max(self.w_min, min(self.w_max, 2.0 * self.w))
            else:
                if (w_count > 5):
                    self.w = max(self.w_min, min(self.w_max, 0.5 * self.w))









