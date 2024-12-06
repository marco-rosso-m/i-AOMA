# from qmc.helpers import *

from scipy.stats import qmc
import numpy as np


class QmcSampler:
    def __init__(self, data, FUNDAMENTAL_FREQUENCY, fs):
        # %% Define qmc sampling limits

        self.fs = fs # sampling frequency
        self.ff = FUNDAMENTAL_FREQUENCY # fundamental mode frequency
        self.NDOFS = data.shape[1] # num of monitored dofs, i.e. tot num of sensors or, in general, channels
        self.Ndata = data.shape[0]

        # SSI time lag or time shift parameter (block rows)
        self.TSmin = np.rint(fs/(2*self.ff)).astype(int)
        self.TSmax = 10*self.TSmin

        # order min and max
        self.nmin = 2*self.NDOFS
        self.nmax = self.TSmax * self.NDOFS

        # time window length
        self.lmin = np.rint(2/self.ff).astype(int) #3*TSmax #int(np.ceil(2/ff))
        self.lmax = self.Ndata

    def Halton(self,_dim=4,_scamble=True,_numsim=1):
        HaltonSamples = qmc.Halton(d=_dim, scramble=_scamble).random(n=_numsim)
        TS= self.TSmin +  np.rint((self.TSmax - self.TSmin) * HaltonSamples[:,0]).astype(int)
        n=self.nmin + np.rint((self.nmax - self.nmin) * HaltonSamples[:,1]).astype(int)
        l= self.lmin +  np.rint((self.lmax - self.lmin) * HaltonSamples[:,2]).astype(int)
        Tt= np.rint(HaltonSamples[:,3] * self.Ndata).astype(int)

        self.qmc_last_sampled_unitary = HaltonSamples
        self.qmc_last_sampled = np.vstack([TS, n, l, Tt]).T

        return self.qmc_last_sampled

    def retrieve_last_qmc_sampled(self,unitary=True):
        if unitary:
            return self.qmc_last_sampled_unitary
        else:
            return self.qmc_last_sampled
