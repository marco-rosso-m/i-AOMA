from scipy.stats import qmc
import numpy as np


class QmcSampler:
    def __init__(self, data, fs, _ff=1):
        # %% Define qmc sampling limits

        self.fs = fs  # sampling frequency
        self.ff = _ff  # fundamental mode frequency
        # num of monitored dofs, i.e. tot num of sensors or, in general, channels
        self.NDOFS = data.shape[1]
        self.Ndata = data.shape[0]

        # SSI time lag or time shift parameter (block rows)
        self.TSmin = np.rint(fs / (2 * self.ff)).astype(int)
        self.TSmax = 10 * self.TSmin

        # order min and max
        self.nmin = 2 * self.NDOFS
        self.nmax = self.TSmax * self.NDOFS

        # time window length
        self.lmin = np.rint(2 / self.ff).astype(int)  # 3*TSmax #int(np.ceil(2/ff))
        self.lmax = self.Ndata

    def print_qmc_limits(self):
        print(
            f"Sampling limits: TSmin={self.TSmin}, TSmax={self.TSmax}, nmin={self.nmin}, nmax={self.nmax}, lmin={self.lmin}, lmax={self.lmax}"
        )

    def Halton(self, _dim=4, _scamble=True, _numsim=1):
        HaltonSamples = qmc.Halton(d=_dim, scramble=_scamble).random(n=_numsim)
        TS = self.TSmin + np.rint(
            (self.TSmax - self.TSmin) * HaltonSamples[:, 0]
        ).astype(int)
        n = self.nmin + np.rint((self.nmax - self.nmin) * HaltonSamples[:, 1]).astype(
            int
        )
        wlen = self.lmin + np.rint(
            (self.lmax - self.lmin) * HaltonSamples[:, 2]
        ).astype(int)
        Tt = np.rint(HaltonSamples[:, 3] * self.Ndata).astype(int)

        self.qmc_last_sampled_unitary = HaltonSamples
        self.qmc_last_sampled = np.vstack([TS, n, wlen, Tt]).T
        # Assuming self.qmc_last_sampled is a 2D array
        self.qmc_last_sampled_list = [
            list(sublist) for sublist in self.qmc_last_sampled
        ]

        return self.qmc_last_sampled_list

    def retrieve_last_qmc_sampled(self, unitary=True):
        if unitary:
            return self.qmc_last_sampled_unitary
        else:
            return self.qmc_last_sampled
