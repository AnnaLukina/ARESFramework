# (c) Anna Lukina 24/05/2018
# objective function

def score(x):
    fit = 0
    for i in (0, x.shape[0]-1):
        fit = fit + max(0, 200 - x[(i,2)])
    return fit


class Objective:
    phi = 0
    T = 10

    def __init__(self, spec, time_bound):
        self.phi = spec
        self.T = time_bound

