# (c) Anna Lukina 24/05/2018
# objective function

from pymatbridge import Matlab


class Objective:
    phi = 0
    T = 10

    def __init__(self, spec, time_bound):
        self.phi = spec
        self.T = time_bound

    def score(self, x):
        fit = 0

        mlab = Matlab (matlab='C:\Program Files\MATLAB\R2017a\bin\matlab')
        mlab.start ()
        res = mlab.run ('gain.m', {'arg1'})

        for i in (0, x.shape[0] - 1):
            fit = fit + max (0, 200 - x[(i, 2)])
        return fit

