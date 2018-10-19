from turtledemo.chaos import plot

import numpy as np
import time

start = time.time()

dim = 8 # problem dimension
Num = 1 # number of agents
# implemented problems to choose from
problems = np.array(['Coverage',
                     'Michalewicz'
                     ])

# define objective function based on given specification

from Objective import*

spec = 10**(-1) # each agent is born with the same global mission
time_bound = 200 # global time constraints
target = -0.99864*dim + 0.30271 # desired global optimum
func = problems[1] # problem choice
obj = Objective(spec,time_bound,func,target)

# define a model

from DynModel import*

swarm = 100 # swarm size
init_lb = 0
init_ub = np.pi
# initial configuration of agents
x0 = np.array(np.random.uniform(init_lb,init_ub,[1,dim]))
dx0 = np.array(np.random.uniform(init_lb,init_ub,[1,dim]))
ddx0 = np.array(np.zeros([1,dim]))
for i in range(1,Num):
    x0 = np.append(x0, np.array(np.random.uniform(init_lb,init_ub,[1,dim])), axis=0)
    dx0 = np.append(dx0, np.array(np.random.uniform(init_lb,init_ub,[1,dim])), axis=0)
    ddx0 = np.append(ddx0, np.zeros([1,dim]), axis=0)
print(x0, "\n", dx0, "\n", ddx0, "\n")
# initialize a model
M = DynModel(obj,x0,dx0,ddx0,Num,dim)
print(M.J)

# call optimizer to find the next optimal action to take

from Optimizer import*
from scipy import linalg as la

horizon = 5
levels = M.J
level = M.J
# initialize optimizer
opt = Optimizer(dim,Num,swarm,M,time_bound)

while M.time < obj.T and level > obj.phi:
    # define lower and upper bounds on the actions based on the velocities of agents
    lb = np.zeros([1,dim])#
    ub = min(la.norm(M.dX[-Num:,:]),init_ub)*np.ones([1,dim])#
    for j in range(1, Num):
        lb = np.append(lb, np.zeros([1,dim]), axis=0)
        ub = np.append(ub, min(la.norm(M.dX[-Num:,:]),init_ub)*np.ones([1,dim]), axis=0)
    # run optimizer for current configuration and constraints
    Optimizer.centralized(opt,horizon, lb, ub)
    if opt.new_level < level:
        # apply the first control inout from the sequence of the best actions to the original model
        DynModel.move(M,np.array(opt.control[-Num:,:]),1,lb,ub)
        # proceed to the next level
        level = opt.new_level
        levels = np.append(levels, level)
        print("new level: ",level)

#import matplotlib.pyplot as plt

#plt.plot(levels)

end = time.time()
print("time: ",end - start)
print("all levels:",levels)
print("solution: ", opt.control)
if level <= obj.phi:
    print("success")


