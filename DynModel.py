# (c) Anna Lukina 24/05/2018
# model of a single agent (as an MDP)

from __future__ import print_function
import numpy as np
from Objective import*


class DynModel:
    # initialization
    X = np.array ([0, 0, 0])
    dX = np.array ([0, 0, 0])
    ddX = np.array ([0, 0, 0])
    time = 0
    Num = 1 # number of agents
    ts = 1 # discrete time steps
    gamma = 0 # discount factor
    # reward function
    J = float ("inf") # initial score for minimization

    def __init__(self, obj, x0, dx0, ddx0, num, dim):
        # state of an MDP
        self.X = x0 # position
        self.dX = dx0 # velocity
        # actions
        self.ddX = ddx0 # acceleration
        self.Num = num # number of agents
        self.dim = dim # dimensionality
        # objective function
        self.obj = obj
        self.J = Objective.score(self.obj,self.X[-self.Num:,:])

    def move(self, ddxi, horizon,lb,ub):
        # proceed to the next state given current time, position, and control input (acceleration)
        self.ddX = np.append(self.ddX, ddxi, axis=0)
        for ph in range (0, horizon):
            self.dX = np.append(self.dX, np.array(self.dX[-self.Num:,:] + self.ddX[-self.Num:,:]), axis=0)
            self.X = np.append(self.X, np.array(self.X[-self.Num:,:] + self.dX[-self.Num:,:]), axis=0)
            self.time = self.time + self.ts
            # check constraints on control variables
            for i in range(0, self.Num - 1):
                for j in range(0, self.dim - 1):
                    if self.X[-self.Num+i, j] < lb[i, j]:
                        self.X[-self.Num+i, j] = lb[i, j]
                    if self.X[-self.Num+i, j] > ub[i, j]:
                        self.X[-self.Num+i, j] = ub[i, j]
        # compute new objective value
        self.J = np.append(self.J, Objective.score(self.obj,self.X[-self.Num:,:]))
