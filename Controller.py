import numpy as np

# define objective function based on given specification

from Objective import*

spec = 0.001 # each agent is born with the same global mission
time_bound = 10 # global time constraints
obj = Objective(spec,time_bound)

# define a model

from Model import*

dim = 3 # problem dimension
Num = 3 # number of agents
swarm = 1 # swarm size
init_lb = 0
init_ub = 100
# initial configuraiton of agents
x0 = np.array(np.random.uniform(init_lb,init_ub,[1,dim]))
dx0 = np.zeros([1,dim])
ddx0 = np.zeros([1,dim])
for i in range(1,Num):
    x0 = np.append(x0, np.array(np.random.uniform(init_lb,init_ub,[1,dim])), axis=0)
    dx0 = np.append(dx0, np.zeros([1,dim]), axis=0)
    ddx0 = np.append(ddx0, np.zeros([1,dim]), axis=0)
print(x0, "\n", dx0, "\n", ddx0, "\n")
# initialize a model
M = Model(obj,x0,dx0,ddx0,Num)
print(M.J)

# call optimizer to find the next optimal action to take

from Optimizer import*

horizon = 1
lb = -10
ub = 10
Optimizer(dim,Num,swarm,M,horizon,lb,ub)