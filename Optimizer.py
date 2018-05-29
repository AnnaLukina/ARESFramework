# call PSO for a given model and horizon

from PSO import*


class Optimizer:
    def __init__(self, dim, num, swarm, model, horizon, lb, ub):
        pso = PSO(dim, num, swarm)
        PSO.step(pso, model, horizon, lb, ub)