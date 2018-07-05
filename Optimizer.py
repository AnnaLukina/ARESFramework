# call PSO for a given model and horizon

from ParticleSwarm import*


class Optimizer:
    def __init__(self, dim, num, swarm, model, time_bound):
        self.pso = ParticleSwarm(dim, num, swarm, model)
        self.model = model
        self.time_bound = time_bound
        self.new_level = model.J
        self.control = np.array(np.zeros([1,dim]))
        self.ph = 0
        self.dim = dim
        self.Num = num

    def centralized(self, horizon, lb, ub):
    # case of full information
        level = self.new_level # current level
        Delta = level / (self.time_bound - self.model.time) / self.dim # dynamical threshold for the Lyapunov function
        print("delta: ",Delta)
        print ("old level: ", level)
        # exploring horizons until a new level is reached
        for Ph in range (1,horizon):
            ParticleSwarm.step(self.pso, self.model, Ph, lb, ub)
            print("best_fit",self.pso.gbest_fit)
            if self.pso.gbest_fit < level and np.fabs(level - self.pso.gbest_fit) >= Delta:
                self.new_level = self.pso.gbest_fit
                self.ph = Ph
                # get the best clone and the sequence of control input that made this clone the best
                self.control = np.append(self.control,np.array(self.pso.models[self.pso.gbest_ind].ddX[-self.Num*self.ph:,:]), axis=0)
                break
