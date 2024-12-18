import numpy as np
from typing import List

from pyoma2.setup import SingleSetup
from pyoma2.algorithms import SSIcov
from pyoma2.algorithms.data.result import SSIResult
from .helper import timeout


class SSHandler:
    def __init__(
        self,
        data: np.ndarray,
        fs: float,
        DecFct: int = 0,
        detrend: bool = False,
        sampled_params: List[int] = [5, 2, -1, -1],
        NumAnal: int = 0,
    ):
        self.NumAnal = NumAnal

        br = sampled_params[0]
        ordmax = sampled_params[1]
        wlen = sampled_params[2]
        tt = sampled_params[3]

        if tt < 0 or tt > data.shape[0]:
            tt = int(data.shape[0] / 2)
        if wlen < 0 or wlen > data.shape[0]:
            wlen = data.shape[0]

        self.data = self.dataslice(data, wlen, tt)
        self.fs = fs

        self.SingleSetup = SingleSetup(self.data, self.fs)

        if DecFct > 0:
            self.SingleSetup.decimate_data(q=DecFct)
            self.fs = fs / DecFct
        if detrend:
            self.SingleSetup.detrend_data()

        # Initialise the algorithms
        self.ssicov = SSIcov(name="SSIcov", br=br, ordmax=ordmax)
        # Add algorithms to the single setup class
        self.SingleSetup.add_algorithms(self.ssicov)

        try:
            self.run_analysis()
            # self.ssicov.result
        except TimeoutError:
            # raise TimeoutError("Timeout of the analysis!")
            print(f"Analysis {self.NumAnal} interrupted: Timeout of the analysis!")
            self.ssicov.result = SSIResult()
        except Exception:
            # raise ValueError("Invalid parameter choice!")
            print(f"Analysis {self.NumAnal} interrupted: Invalid parameter choice!")
            self.ssicov.result = SSIResult()

        # _, _ = self.ssicov.plot_stab(freqlim=(0,self.SingleSetup.fs/2),hide_poles=False)
        # mplcursors.cursor()

        # # plot frequecy-damping clusters for SSI
        # _, _ = self.ssicov.plot_cluster(freqlim=(0,self.SingleSetup.fs/2))
        # mplcursors.cursor()

    def dataslice(self, data: np.ndarray, wlen: float, tt: int):
        if tt - int(wlen / 2) < 0:
            Sliceddata = data[0 : int(wlen), :]
        elif tt + int(wlen / 2) > data.shape[0]:
            Sliceddata = data[-int(wlen) :, :]
        else:
            Sliceddata = data[tt - int(wlen / 2) : tt + int(wlen / 2), :]

        return Sliceddata

    @timeout()
    def run_analysis(self):
        self.SingleSetup.run_all()
