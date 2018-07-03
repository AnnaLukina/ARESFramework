from turtledemo.chaos import plot

import numpy as np
import math

# define objective function based on given specification

from Objective import*

spec = 0.1 # each agent is born with the same global mission
time_bound = 100 # global time constraints
obj = Objective(spec,time_bound,'Michalewicz')

# define a model

from DynModel import*

dim = 2 # problem dimension
Num = 1 # number of agents
swarm = 100 # swarm size
init_lb = 0
init_ub = np.pi
# initial configuration of agents
x0 = np.array(np.random.uniform(init_lb,init_ub,[1,dim]))
dx0 = np.zeros([1,dim])
ddx0 = np.zeros([1,dim])
for i in range(1,Num):
    x0 = np.append(x0, np.array(np.random.uniform(init_lb,init_ub,[1,dim])), axis=0)
    dx0 = np.append(dx0, np.zeros([1,dim]), axis=0)
    ddx0 = np.append(ddx0, np.zeros([1,dim]), axis=0)
print(x0, "\n", dx0, "\n", ddx0, "\n")
# initialize a model
M = DynModel(obj,x0,dx0,ddx0,Num)
print(M.J)

# call optimizer to find the next optimal action to take

from Optimizer import*
from scipy import linalg as la

horizon = 7
levels = M.J
level = M.J
# initialize optimizer
opt = Optimizer(dim,Num,swarm,M,time_bound)

while M.time < obj.T and level > obj.phi:
    # define lower and upper bounds on the actions based on the velocities of agents
    lb = np.zeros([1,dim])#
    ub = max(la.norm(M.dX),np.pi)*np.ones([1,dim])#
    for j in range(1, Num):
        lb = np.append(lb, np.zeros([1,dim]), axis=0)
        ub = np.append(ub, max(la.norm(M.dX),np.pi)*np.ones([1,dim]), axis=0)
    # run optimizer for current configuration and constraints
    Optimizer.centralized(opt,horizon, lb, ub)
    if opt.new_level < level:
        DynModel.move(M,opt.control,opt.ph)
        level = M.J
        levels = np.append(levels, M.J)
        print("new level: ",level)

#import matplotlib.pyplot as plt

#plt.plot(levels)
print("all levels:",levels)
if level <= obj.phi:
    print("success")
