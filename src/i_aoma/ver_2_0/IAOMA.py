import numpy as np
import copy
import logging
import os

from .pyoma2.setup import SingleSetup
from .pyoma2.algorithms import SSIcov

from .IAOMAPhase1 import IAOMAPhase1
from .IAOMAPhase2 import IAOMAPhase2

# Erase the content of the log file if it exists
log_file = "iaoma_run.log"
if os.path.exists(log_file):
    open(log_file, "w").close()

# Set up logging
logging.basicConfig(
    filename="iaoma_run.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode="w+",
)  # Use 'w' for overwrite mode, 'a' for append mode


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
        DecFct: int = 0,
        detrend: bool = False,
        output_path: str = os.getcwd() + os.sep + "IAOMA_Results",
    ):
        # Initialize shared attributes
        print("Initialize i-AOMA...")
        logging.info("Initialize i-AOMA...")

        # Create output folder if it does not exist
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.SingleSetup = SingleSetup(data, fs=fs)
        if detrend:
            print("Detrending data...")
            logging.info("Detrending data...")
            self.SingleSetup.detrend_data()

        if DecFct > 1:
            print("Decimating data...")
            logging.info("Decimating data...")
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
        self.ordmin = 0  # 2 * self.NDOFS
        self.ordmax = self.brmax * self.NDOFS

        # time window length
        self.wlenmin = np.rint(2 / ff).astype(int)  # 3*TSmax #int(np.ceil(2/ff))
        self.wlenmax = self.Ndata

    def print_qmc_sampling_limits(self):
        print(
            f"Sampling limits: br_min={self.brmin}, br_max={self.brmax}, order_min={self.ordmin}, order_max={self.ordmax}, window_length_min={self.wlenmin}, window_length_max={self.wlenmax}"
        )

    def run_phase1(
        self,
        NsimPh1: int = 1,
        n_jobs: int = -1,
        timeout_seconds: int = 30,
        Nsim_batch: int = 1,
        set_seed=None,
        plt_resolution: dict = {"freq": 0.5, "damp": 0.001, "order": 1},
        plt_stab_diag_backup: bool = False,
    ):
        """
        Creates an instance of IAOMAPhase1 and executes Phase 1 operations.
        """
        print("Starting Phase 1...")
        logging.info("Starting Phase 1...")
        self.phase1 = IAOMAPhase1(self)
        fig, ax = self.phase1.loop_phase1_operations(
            NsimPh1,
            n_jobs,
            timeout_seconds,
            Nsim_batch,
            set_seed,
            plt_resolution,
            plt_stab_diag_backup,
        )
        # return self.results  # Return phase1 object for further use
        return fig, ax

    def dump_phase1_to_file(self, output_path: str = None):
        # TODO: implement checks before saving or at least a try/except block
        if output_path is not None:
            self.phase1._dump_metadata_to_file_phase1(output_path)
            self.phase1._dump_results_to_file_phase1(output_path)
        elif hasattr(self.phase1, "output_path_phase1"):
            self.phase1._dump_metadata_to_file_phase1(self.phase1.output_path_phase1)
            self.phase1._dump_results_to_file_phase1(self.phase1.output_path_phase1)
        else:
            print("No results to save. Run Phase 1 first.")
            logging.info("No results to save. Run Phase 1 first.")
            return None

    def load_phase1_from_file(self, phase1_files: list):
        """
        Load Phase 1 results from files.
        """
        # TODO: implement checks on file before loading or at least a try/except block
        print("Loading Phase 1 results...")
        logging.info("Loading Phase 1 results...")
        self.phase1 = IAOMAPhase1(self)
        self.phase1._load_metadata_from_file_phase1(phase1_files[0])
        self.phase1._load_results_from_file_phase1(phase1_files[1])

    # def plot_phase1_overlapped_stab_diag(self, method: str = "density"):
    #     """
    #     Plot the results of Phase 1: overlapped stabilization diagram.
    #     """
    #     fig, ax = self.phase1.plot_overlap_stab_diag(method=method)

    #     return fig, ax

    # def plot_phase1_overlapped_cluster_diag(self, method: str = "density"):
    #     """
    #     Plot the results of Phase 1: overlapped damping cluster diagram.
    #     """
    #     fig, ax = self.phase1.plot_overlap_freq_damp_cluster(method=method)
    #     print("Plotting Phase 1 results: overlapped damping cluster diagram...")
    #     logging.info("Plotting Phase 1 results: overlapped damping cluster diagram...")
    #     return fig, ax

    def phase2_start(self):
        if hasattr(self, "phase1"):  # if phase 1 already run
            print("Starting Phase 2...")
            logging.info("Starting Phase 2...")
            self.phase2 = IAOMAPhase2()
            self.phase2.check_convergence()

    def run_phase2(self, phase1_object):
        """
        Creates an instance of IAOMAPhase2 using Phase 1 results.
        """
        if hasattr(self, "phase1"):  # if phase 1 already run
            print("Starting Phase 2...")
            logging.info("Starting Phase 2...")
            self.phase2 = IAOMAPhase2(phase1_object)
            self.phase2.loop_phase2_operations()
            # return phase2  # Return phase2 object for further use
        else:
            print("Phase 1 not run yet. Run Phase 1 first.")
            logging.info("Phase 1 not run yet. Run Phase 1 first.")
            return None

    # def dump_results_to_file(self):
    #     """
    #     Save the results of Phase 1 and Phase 2 in a file.
    #     """
    #     if hasattr(self, 'phase2'):
    #         # if phase 2 already run, then save both phase 1 and 2
    #         with shelve.open(self.output_path+os.sep+'backup_shelve') as db:
    #             db['phase1'] = self.phase1
    #             db['phase2'] = self.phase2
    #     elif hasattr(self, 'phase1'):
    #         # if phase 1 already run, then save both phase 1 and 2
    #         with shelve.open(self.output_path+os.sep+'backup_shelve') as db:
    #             db['phase1'] = self.phase1
    #     else:
    #         print("No results to save. Run Phase 1 and Phase 2 first.")
    #         logging.info("No results to save. Run Phase 1 and Phase 2 first.")
    #         return None

    # def dump_results_to_file_pickle(self):
    #     """
    #     Save the results of Phase 1 and Phase 2 in a file.
    #     """
    #     if hasattr(self, 'phase2'):
    #         # if phase 2 already run, then save both phase 1 and 2
    #         with open(self.output_path+os.sep+'backup_pickle_phase1.pkl.gz', 'wb') as backup_file:
    #             pickle.dump(self.phase1, backup_file)
    #         with shelve.open(self.output_path+os.sep+'backup_pickle_phase2', 'wb') as backup_file:
    #             pickle.dump(self.phase2, backup_file)
    #     elif hasattr(self, 'phase1'):
    #         # if phase 1 already run, then save both phase 1 and 2
    #         with open(self.output_path+os.sep+'backup_pickle_phase1.pkl.gz', 'wb') as backup_file:
    #             pickle.dump(self.phase1, backup_file)
    #     else:
    #         print("No results to save. Run Phase 1 and Phase 2 first.")
    #         logging.info("No results to save. Run Phase 1 and Phase 2 first.")
    #         return None
