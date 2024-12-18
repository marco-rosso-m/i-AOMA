# -*- coding: utf-8 -*-
"""

Author:
Marco Martino Rosso
"""

import numpy as np
from joblib import Parallel, delayed

# from pyoma2.functions.plot import plot_mac_matrix
from .DataHandler import SingleSetupHandler
from .QmcSampler import QmcSampler

# plt.plot([0, 0])
# plt.close()


class IAOMASingleSetup:
    def __init__(
        self,
        data: np.ndarray,
        fs: float,
        NsimPh1: int,
        ff: float = 0.5,
        DecFct: int = 0,
        detrend: bool = False,
    ):
        self.qmc = QmcSampler(data, fs, _ff=ff)
        self.sampledpar = self.qmc.Halton(
            _numsim=NsimPh1
        )  # it samples an array of [time shift, max order, window lenght, time target]

        # self.data_handler = SingleSetupHandler(data, fs, DecFct = DecFct, detrend = detrend)#, sampled_params=self.sampledpar[0])

        # Parallelize the creation of SingleSetupHandler instances
        self.data_handlers = Parallel(n_jobs=-1)(
            delayed(SingleSetupHandler)(
                data,
                fs,
                DecFct=DecFct,
                detrend=detrend,
                sampled_params=params,
                NumAnal=NumAnal,
            )
            for NumAnal, params in enumerate(self.sampledpar)
        )

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
