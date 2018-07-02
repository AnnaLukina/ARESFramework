# call PSO for a given model and horizon

from ParticleSwarm import*


class Optimizer:
    def __init__(self, dim, num, swarm, model, time_bound):
        self.pso = ParticleSwarm(dim, num, swarm)
        self.model = model
        self.time_bound = time_bound
        self.new_level = 0
        self.control = 0
        self.ph = 0

    def centralized(self, horizon, lb, ub):
    # case of full information
        level = self.model.J # current level
        Delta = level / (self.time_bound - self.model.time) # dynamical threshold for the Lyapunov function
        print ("level", level)
        # exploring horizons until a new level is reached
        for Ph in range (0,horizon):
            ParticleSwarm.step(self.pso, self.model, Ph, lb, ub)
            print("best_fit",self.pso.gbest_fit)
            if level - self.pso.gbest_fit > Delta:
                self.new_level = self.pso.gbest_fit
                self.control = self.pso.gbest
                self.ph = Ph
                break
