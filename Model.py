# (c) Anna Lukina 24/05/2018
# model of a single agent (as an MDP)

from __future__ import print_function
import numpy as np
from Objective import*


class Model:
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

    def __init__(self, obj, x0, dx0, ddx0, num):
        # state of an MDP
        self.X = x0 # position
        self.dX = dx0 # velocity
        # actions
        self.ddX = ddx0 # acceleration
        self.Num = num # number of agents
        # objective function
        self.obj = obj
        self.J = score(self.X)

    def move(self, ddxi, horizon):
        # proceed to the next state given current time, position, and control input (acceleration)
        self.ddX = ddxi
        for ph in range (0, horizon):
            self.dX = self.dX + self.ddX
            self.X = self.X + self.dX
            self.time = self.time + self.ts
        self.J = score(self.X)
