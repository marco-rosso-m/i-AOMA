# -*- coding: utf-8 -*-
"""

Author:
Marco Martino Rosso
"""

import numpy as np
# from scipy.stats import qmc
# from joblib import Parallel, delayed

# # from pyoma2.functions.plot import plot_mac_matrix
# from .helper import timeout

from .IAOMAPhase1 import IAOMAPhase1

# plt.plot([0, 0])
# plt.close()


class IAOMA:
    def __init__(self, data: np.ndarray, fs: float, ff: float = 0.5):
        self.data = data
        self._qmc_sample_init(data, fs, ff)
        self.print_qmc_limits()

        self.iaomaphase1 = IAOMAPhase1()

    def _qmc_sample_init(self, data, fs, _ff):
        self.fs = fs  # sampling frequency
        self.ff = _ff  # fundamental mode frequency
        # num of monitored dofs, i.e. tot num of sensors or, in general, channels
        self.NDOFS = data.shape[1]
        self.Ndata = data.shape[0]

        # SSI time lag or time shift parameter (block rows)
        self.brmin = np.rint(fs / (2 * self.ff)).astype(int)
        self.brmax = 10 * self.brmin

        # order min and max
        self.nmin = 2 * self.NDOFS
        self.nmax = self.brmax * self.NDOFS

        # time window length
        self.wlenmin = np.rint(2 / self.ff).astype(int)  # 3*TSmax #int(np.ceil(2/ff))
        self.wlenmax = self.Ndata

    def print_qmc_limits(self):
        print(
            f"Sampling limits: TSmin={self.brmin}, TSmax={self.brmax}, nmin={self.nmin}, nmax={self.nmax}, lmin={self.wlenmin}, lmax={self.wlenmax}"
        )

    def run_phase1(
        self, NsimPh1: int, DecFct: int = 0, detrend: bool = False, n_jobs: int = -1
    ):
        pass
        # IAOMAphase1 = IAOMAPhase1()

    #     if n_jobs == 0:
    #         for NumAnal in range( NsimPh1 + 1 ):
    #             results = self._phase1_iter(
    #                                         DecFct=DecFct,
    #                                         detrend=detrend,
    #                                         NumAnal=NumAnal,
    #                                     )
    #     else :
    #         # Parallelize the creation of SingleSetupHandler instances
    #         results = Parallel(n_jobs=n_jobs)(
    #             delayed(self._phase1_iter)(
    #                 DecFct=DecFct,
    #                 detrend=detrend,
    #                 NumAnal=NumAnal,
    #             )
    #             for NumAnal in range( NsimPh1 + 1 )
    #         )

    # def _phase1_iter(self, DecFct, detrend, NumAnal):

    #     missing_good_results = 1
    #     bad_sampled_par = []
    #     while missing_good_results:
    #         try:
    #             sampled_par = self._run_phase1_iter(DecFct, detrend, NumAnal)
    #             missing_good_results = 0
    #         except TimeoutError:
    #             print(f"Analysis {NumAnal} interrupted: Timeout of the analysis!")
    #             bad_sampled_par.append(sampled_par)
    #         except Exception:
    #             print(f"Analysis {NumAnal} interrupted: Invalid parameter choice!")
    #             bad_sampled_par.append(sampled_par)

    #     return [[NumAnal, sampled_par], [bad_sampled_par]]
    #     # [qmc_sampled_set, pyoma2SS] = self._pre_run_phase1(data, DecFct, detrend)
    #     # ssicov = self._run_SSIcov(pyoma2SS)
    #     # return [[NumAnal, qmc_sampled_set, ssicov.result, ssicov.runparams], ]

    # def _run_phase1_iter(self):
    #     self.NumAnal = NumAnal

    #     br = sampled_params[0]
    #     ordmax = sampled_params[1]
    #     wlen = sampled_params[2]
    #     tt = sampled_params[3]

    #     if tt < 0 or tt > data.shape[0]:
    #         tt = int(data.shape[0] / 2)
    #     if wlen < 0 or wlen > data.shape[0]:
    #         wlen = data.shape[0]

    #     self.data = self.dataslice(data, wlen, tt)
    #     self.fs = fs

    #     self.SingleSetup = SingleSetup(self.data, self.fs)

    # @timeout()
    # def _run_SSIcov(self):
    #     pass

    # def _qmc_HaltonSampling(self, _dim=4, _scamble=True, _numsim=1):
    #     HaltonSamples = qmc.Halton(d=_dim, scramble=_scamble).random(n=_numsim)
    #     TS = self.TSmin + np.rint(
    #         (self.TSmax - self.TSmin) * HaltonSamples[:, 0]
    #     ).astype(int)
    #     n = self.nmin + np.rint((self.nmax - self.nmin) * HaltonSamples[:, 1]).astype(
    #         int
    #     )
    #     wlen = self.lmin + np.rint(
    #         (self.lmax - self.lmin) * HaltonSamples[:, 2]
    #     ).astype(int)
    #     Tt = np.rint(HaltonSamples[:, 3] * self.Ndata).astype(int)

    #     self.qmc_last_sampled_unitary = HaltonSamples
    #     self.qmc_last_sampled = np.vstack([TS, n, wlen, Tt]).T
    #     # Assuming self.qmc_last_sampled is a 2D array
    #     self.qmc_last_sampled_list = [
    #         list(sublist) for sublist in self.qmc_last_sampled
    #     ]

    #     return self.qmc_last_sampled_list

    # self.qmc.print_qmc_limits()

    # self.sampledpar = self.qmc.Halton(
    #     _numsim=NsimPh1
    # )  # it samples an array of [time shift, max order, window lenght, time target]

    # # self.data_handler = SingleSetupHandler(data, fs, DecFct = DecFct, detrend = detrend)#, sampled_params=self.sampledpar[0])

    # # Parallelize the creation of SingleSetupHandler instances
    # self.data_handlers = Parallel(n_jobs=-1)(
    #     delayed(SingleSetupHandler)(
    #         data,
    #         fs,
    #         DecFct=DecFct,
    #         detrend=detrend,
    #         sampled_params=params,
    #         NumAnal=NumAnal,
    #     )
    #     for NumAnal, params in enumerate(self.sampledpar)
    # )

    # def def_geo(self, *arg):
    #     """
    #     arg contains path to excel file for defining geometry according to PyOMA2 standard
    #     _geo1,_geo2
    #     """
    #     try:
    #         if len(arg) == 1:
    #             self.ss.def_geo1_by_file(arg[0])
    #         else:
    #             self.ss.def_geo1_by_file(arg[0])
    #             self.ss.def_geo2_by_file(arg[1])
    #     except Exception:
    #         raise ValueError("Invalid geometry type!")
