import numpy as np
import copy

from .pyoma2.setup import SingleSetup
from .pyoma2.algorithms import SSIcov

from .IAOMAPhase1 import IAOMAPhase1
from .IAOMAPhase2 import IAOMAPhase2

from memory_profiler import profile


class IAOMA:
    """
    Parent class implementing high-level methods for the user to call.
    Contains shared attributes.
    """

    def __init__(
        self,
        data: np.ndarray,
        fs: float,
        ff: float = 1.0,
        DecFct=0,
        detrend=False,
    ):
        # Initialize shared attributes

        self.SingleSetup = SingleSetup(data, fs=fs)
        if detrend:
            print("Detrending data...")
            self.SingleSetup.detrend_data()

        if DecFct > 1:
            print("Decimating data...")
            self.SingleSetup.decimate_data(q=DecFct)

        self.SingleSetup._initial_data = copy.deepcopy(self.SingleSetup.data)
        self.SingleSetup._initial_fs = self.SingleSetup.fs

        ssicov = SSIcov(name="SSIcov", br=50, ordmax=50, method="cov_R")
        # Add algorithms to the single setup class
        self.SingleSetup.add_algorithms(ssicov)

        # num of monitored dofs, i.e. tot num of sensors or, in general, channels
        self.NDOFS = data.shape[1]
        self.Ndata = data.shape[0]

        self._qmc_sampling_limits_init(ff)
        self.print_qmc_sampling_limits()

    def _qmc_sampling_limits_init(self, ff):
        # SSI time lag or time shift parameter (block rows)
        self.brmin = np.rint(self.SingleSetup.fs / (2 * ff)).astype(int)
        self.brmax = 10 * self.brmin

        # order min and max (min was not used for the moment)
        self.ordmin = 2 * self.NDOFS
        self.ordmax = self.brmax * self.NDOFS

        # time window length
        self.wlenmin = np.rint(2 / ff).astype(int)  # 3*TSmax #int(np.ceil(2/ff))
        self.wlenmax = self.Ndata

    def print_qmc_sampling_limits(self):
        print(
            f"Sampling limits: br_min={self.brmin}, br_max={self.brmax}, order_min={self.ordmin}, order_max={self.ordmax}, window_length_min={self.wlenmin}, window_length_max={self.wlenmax}"
        )

    @profile
    def run_phase1(
        self, NsimPh1: int, n_jobs: int = -1, timeout_seconds: int = 30, set_seed=None
    ):
        """
        Creates an instance of IAOMAPhase1 and executes Phase 1 operations.
        """
        print("Starting Phase 1...")
        phase1 = IAOMAPhase1(self)
        self.results = phase1.loop_phase1_operations(
            NsimPh1, n_jobs, timeout_seconds, set_seed
        )
        return self.results  # Return phase1 object for further use

    def run_phase2(self, phase1_object):
        """
        Creates an instance of IAOMAPhase2 using Phase 1 results.
        """
        print("Starting Phase 2...")
        phase2 = IAOMAPhase2(phase1_object)
        phase2.loop_phase2_operations()
        return phase2  # Return phase2 object for further use


# class IAOMAPhase1:
#     """
#     Child class for Phase 1 operations.
#     Inherits attributes from IAOMA but not its methods.
#     """
#     def __init__(self, iaoma):
#         # Inherit attributes from IAOMA
#         self.data = iaoma.data
#         self.fs = iaoma.fs
#         self.ff = iaoma.ff
#         self.br = iaoma.br
#         self.wlen = iaoma.wlen
#         self.tt = iaoma.tt
#         self.ordmax = iaoma.ordmax

#     def loop_phase1_operations(self, NsimPh1, DecFct, detrend, n_jobs):
#         """
#         Loop operations until collecting NsimPh1 results.
#         """
#         print("Sampling parameters and running Phase 1 operations...")
#         # Example of operations
#         self.qmc_sample()
#         self.detrend_and_decimate()
#         self.SSICov()
#         print("Phase 1 operations completed.")

#     def qmc_sample(self):
#         print("QMC Sampling...")

#     def detrend_and_decimate(self):
#         print("Detrending and Decimating...")

#     def SSICov(self):
#         print("SSICov analysis...")


# class IAOMAPhase2(IAOMAPhase1):
#     """
#     Child class for Phase 2 operations.
#     Inherits attributes from IAOMA and methods from IAOMAPhase1.
#     Implements new functionality specific to Phase 2.
#     """
#     def __init__(self, phase1_object):
#         # Inherit attributes from IAOMAPhase1 (and IAOMA indirectly)
#         super().__init__(phase1_object)

#     def loop_phase2_operations(self):
#         """
#         Loop operations until convergence.
#         """
#         print("Running Phase 2 operations with convergence checks...")
#         self.qmc_sample()  # Reuse method from IAOMAPhase1
#         self.detrend_and_decimate()
#         self.SSICov()
#         self.check_convergence()
#         print("Phase 2 operations completed.")

#     def check_convergence(self):
#         print("Checking convergence of results...")
