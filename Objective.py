# (c) Anna Lukina 24/05/2018
# objective function

#from pymatbridge import Matlab
import numpy as np


class Objective:
    phi = 0
    T = 10

    def __init__(self, spec, time_bound, func):
        self.phi = spec
        self.T = time_bound
        self.func = func

    def score(self, x):
        #mlab = Matlab (matlab='C:\Program Files\MATLAB\R2017a\bin\matlab')
        #mlab.start ()
        #res = mlab.run ('gain.m', {'arg1'})

        # benchmark function for global optimization
        if self.func == 'Michalewicz':
            fit = 0
            for i in (0,x.shape[0]-1):
                fit = fit + np.sum(-np.sin(x[i,:])*(np.sin((i+1)*x[i,:]**2/np.pi))**20)
            fit = np.fabs(-0.99999981-fit)

        # altitude function for drone coverage task
        if self.func == 'Coverage':
            fit = 0
            for i in (0, x.shape[0] - 1):
                fit = fit + max (0, 200 - x[(i, 2)])

        return fit

