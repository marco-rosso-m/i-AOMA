import numpy as np
import copy
import logging
import os
import pickle
import glob
from sklearn.ensemble import RandomForestClassifier

from pyoma2.setup import SingleSetup
from pyoma2.algorithms import SSIcov

from .IAOMAPhase1 import IAOMAPhase1
from .IAOMAPhase2 import IAOMAPhase2
from .IAOMAResults import IAOMAResults

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

    # IAOMA INSTANCE INITIALIZATION AND PREPROCESSING
    def __init__(
        self,
        data: np.ndarray,
        fs: float,
        ff: float = 1.0,
        output_path: str = os.getcwd() + os.sep + "IAOMA_Results",
        plt_resolution: dict = {
            "freq": 0.5,
            "damp": 0.001,
            "order": 1,
            "freq_kde": 0.05,
        },
        modal_cluster_resolution: dict = {
            "fr_diff_threshold": 0.10,
            "signifiant_digits_cluster_fr_keys": 2,
            "MAC_lower_threshold_to_separate_clusters": 0.95,
        },
    ):
        """
        Initialize the IAOMA class.
        """
        # Initialize shared attributes
        print("Initialize i-AOMA...")
        logging.info("Initialize i-AOMA...")

        # Create output folder if it does not exist
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.SingleSetup = SingleSetup(data, fs=fs)
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

        self.plt_resolution = plt_resolution
        self.running_phase = ""
        self.NsimPh_max = 1
        self.Nsim_batch = 1
        self.modal_cluster_resolution = modal_cluster_resolution

        self.clf_model = None
        self.convergence_threshold = 0.01  # track relative differences covariance matrix within +-1% for acceptable shifting convergence band rule (ASCBR) default value

        self.Results = IAOMAResults(self)

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

    def manual_update_sampling_limits(self, **kwargs):
        """
        kwargs: brmin, brmax, ordmax, wlenmin, wlenmax
        """
        if "brmin" in kwargs:
            self.brmin = kwargs["brmin"]
        if "brmax" in kwargs:
            self.brmax = kwargs["brmax"]
        if "ordmax" in kwargs:
            self.ordmax = kwargs["ordmax"]
        if "wlenmin" in kwargs:
            self.wlenmin = kwargs["wlenmin"]
        if "wlenmax" in kwargs:
            self.wlenmax = kwargs["wlenmax"]

        self.Results = IAOMAResults(self)

    def preprocess_data(self, detrend: bool = False, DecFct: int = 0):
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
        self.Results.SingleSetup = self.SingleSetup

    # PHASE 1 INITIALIZATION AND RUN
    def run_phase1(
        self,
        NsimPh_max: int = 1,
        n_jobs: int = -1,
        timeout_seconds: int = 30,
        Nsim_batch: int = 1,
        set_seed=None,
        plt_stab_diag_backup: bool = False,
        progressive_plot_flag: bool = True,
    ):
        """
        Creates an instance of IAOMAPhase1 and executes Phase 1 operations.
        """

        self.running_phase = "Phase1"
        self.NsimPh_max = NsimPh_max
        self.Nsim_batch = Nsim_batch

        print("Starting Phase 1...")
        logging.info("Starting Phase 1...")

        self.phase1 = IAOMAPhase1(self)

        fig, ax = self.phase1.loop_phase_operations(
            n_jobs,
            timeout_seconds,
            set_seed,
            plt_stab_diag_backup,
            progressive_plot_flag,
        )

        return fig, ax

    # LEARNING PHASE BETWEEN PHASE 1 AND PHASE 2

    # TRAINING INTELLIGENT CORE RF MODEL
    def rf_intelligent_core_training(self, ICTHRESH: float = 0.2):
        logging.info(
            "====================================================================================================="
        )
        if len(self.Results.sim_results) == 0 or np.all(
            np.isnan(
                [
                    self.Results.sim_results[ii]["IC"][1]
                    for ii in range(len(self.Results.sim_results))
                ]
            )
        ):
            print(
                "Error: no results to train the intelligent core of IAOMA. Please Run Phase 1 first or load results from backup files."
            )
            logging.error(
                "Error: no results to train the intelligent core of IAOMA. Please Run Phase 1 first or load results from backup files."
            )
        elif len(self.Results.discarded_qmc_samples) == 0:
            print(
                "Warning: Be aware that there are No discarded qmc samples to properly train the intelligent core RF model."
            )
            logging.error(
                "Warning: Be aware that there are No discarded qmc samples to properly train the intelligent core RF model."
            )
            # return None
            # TODO: it could be useful to leave the user the possibility
            # to retrain the RF model when new discarded qmc samples are
            # found during phase 2 (e.g. every batch....)

        ic_values, db_qmc_samples = [], []
        for ii, res in enumerate(self.Results.sim_results):
            ic_values.append(res["IC"][1])
            db_qmc_samples.append(
                res["qmc_sample"][:, 1]
            )  # unitary qmc admissible samples
        for ii, sample in enumerate(self.Results.discarded_qmc_samples):
            unitary_sample = sample[1]
            ic_values.append(0.0)
            db_qmc_samples.append(unitary_sample)  # unitary qmc discarded samples

        db_qmc_samples = np.array(db_qmc_samples)
        ic_values = np.array(ic_values)

        target_variables = (ic_values > ICTHRESH).astype("uint8")

        self.clf_model = RandomForestClassifier()
        # n_estimators int, default=100 The number of trees in the forest.
        # criterion{“gini”, “entropy”, “log_loss”}, default=”gini”
        # max_depthint, default=None The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.

        self.clf_model.fit(db_qmc_samples, target_variables)

        # from .helper import get_qmc_sample
        # qmc_sample, qmc_sample_unitary = get_qmc_sample(
        #     self.qmc_limits
        # )
        # y_pred = self.clf_model.predict( np.array(qmc_sample_unitary) )

        # TODO: in this moment, there are no discarded qmc samples, therefore, the RF is useless, because it will accept everything maybe?
        # is it possible to explore the density plot verifying if the RF is able to get anyway some most probable region or not?
        print("Intelligent Core RF trained... ")
        logging.error("Intelligent Core RF trained... ")
        return self.clf_model

    # PHASE 2 INITIALIZATION AND RUN
    def run_phase2(
        self,
        NsimPh_max: int = 1,
        n_jobs: int = -1,
        timeout_seconds: int = 30,
        Nsim_batch: int = 1,
        convergence_threshold: float = 0.01,  # track relative differences covariance matrix within +-2% for acceptable shifting convergence band rule (ASCBR)
        set_seed=None,
        plt_stab_diag_backup: bool = False,
        progressive_plot_flag: bool = True,
    ):
        if self.clf_model is not None:
            self.running_phase = "Phase2"
            self.NsimPh_max = NsimPh_max
            self.Nsim_batch = Nsim_batch
            self.convergence_threshold = convergence_threshold

            print("Starting Phase 2...")
            logging.info("Starting Phase 2...")

            self.phase2 = IAOMAPhase2(self)

            fig, ax = self.phase2.loop_phase_operations(
                n_jobs,
                timeout_seconds,
                set_seed,
                plt_stab_diag_backup,
                progressive_plot_flag,
            )

            return fig, ax
        else:
            print(
                "Phase 1 intelligent core not trained yet. Train the Phase 1 intelligent core first."
            )
            logging.info(
                "Phase 1 intelligent core not trained yet. Train the Phase 1 intelligent core first."
            )
            return None

    def dump_all_results_to_file(
        self, output_path: str = os.getcwd() + os.sep + "IAOMA_Results"
    ):
        self.dump_sim_results_to_file(output_path + os.sep + "Backup_All_Sim_Results")
        self.dump_sim_results_to_file(output_path + os.sep + "Backup_All_Sim_Results")

    def dump_sim_results_to_file(
        self,
        output_path: str = os.getcwd()
        + os.sep
        + "IAOMA_Results"
        + os.sep
        + "Backup_All_Sim_Results",
    ):
        for file in glob.glob(output_path + os.sep + "*.pkl"):
            os.remove(file)
        with open(
            output_path + os.sep + f"{len(self.Results.sim_results):d}.pkl", "wb"
        ) as backup_file:
            pickle.dump(self.Results.sim_results, backup_file)

    def dump_discarded_qmc_samples_to_file(
        self,
        output_path: str = os.getcwd()
        + os.sep
        + "IAOMA_Results"
        + os.sep
        + "Backup_Discarded_Samples",
    ):
        for file in glob.glob(output_path + os.sep + "*.pkl"):
            os.remove(file)
        with open(
            output_path + os.sep + f"{len(self.Results.sim_results):d}.pkl", "wb"
        ) as backup_file:
            pickle.dump(self.Results.discarded_qmc_samples, backup_file)

    def load_sim_results_from_a_file(
        self, file_path: str = os.getcwd() + os.sep + "IAOMA_Results"
    ):
        pass

    # def dump_phase1_to_file(self, output_path: str = None):
    #     # TODO: implement checks before saving or at least a try/except block
    #     if output_path is not None:
    #         self.phase1._dump_metadata_to_file_phase1(output_path)
    #         self.phase1._dump_results_to_file_phase1(output_path)
    #     elif hasattr(self.phase1, "output_path_phase1"):
    #         self.phase1._dump_metadata_to_file_phase1(self.phase1.output_path_phase1)
    #         self.phase1._dump_results_to_file_phase1(self.phase1.output_path_phase1)
    #     else:
    #         print("No results to save. Run Phase 1 first.")
    #         logging.info("No results to save. Run Phase 1 first.")
    #         return None

    # def load_phase1_from_file(self, phase1_files: list):
    #     """
    #     Load Phase 1 results from files.
    #     """
    #     # TODO: implement checks on file before loading or at least a try/except block
    #     print("Loading Phase 1 results...")
    #     logging.info("Loading Phase 1 results...")
    #     self.phase1 = IAOMAPhase1(self)
    #     self.phase1._load_metadata_from_file_phase1(phase1_files[0])
    #     self.phase1._load_results_from_file_phase1(phase1_files[1])

    def run_phase2_old(
        self,
        NsimPh2: int = 1,
        n_jobs: int = -1,
        timeout_seconds: int = 30,
        Nsim_batch: int = 1,
        Nsim_batch_conv_check: int = 1,
        set_seed=None,
        beta_distribution_percentile: float = 0.99,  # percentile to be used to define the prominence threshold
        plt_resolution: dict = {"freq": 0.5, "damp": 0.001, "order": 1},
        fr_diff_threshold: float = 0.10,  # Determine precision if two frequencies can be considered different cluster based on a threshold
        signifiant_digits_cluster_fr_keys: int = 2,  # Determine the number of significant digits to round the frequency cluster key
        plot_clusters_damp_kde: bool = True,
        plt_stab_diag_backup: bool = False,
    ):
        if hasattr(
            self.phase1, "clf_model"
        ):  # if phase 1 intelligent core already trained
            if hasattr(self, "phase1"):  # if phase 1 already run
                print("Starting Phase 2...")
                logging.info("Starting Phase 2...")
                self.phase2 = IAOMAPhase2(
                    self,
                    self.phase1.Nsim_batch,
                    self.phase1.KDEPROMINENCE,
                    self.phase1.bw,
                    self.phase1.clusters,
                    self.phase1.clf_model,
                    self.phase1.sim_results,
                    self.phase1.discarded_qmc_samples,
                    self.phase1.NsimPh1,
                )
                fig, ax = self.phase2.loop_phase2_operations(
                    NsimPh2,
                    n_jobs,
                    timeout_seconds,
                    Nsim_batch,
                    Nsim_batch_conv_check,
                    set_seed,
                    beta_distribution_percentile,
                    plt_resolution,
                    fr_diff_threshold,
                    signifiant_digits_cluster_fr_keys,
                    plot_clusters_damp_kde,
                    plt_stab_diag_backup,
                )
                return fig, ax
            else:
                print("Phase 1 not run yet. Run Phase 1 first.")
                logging.info("Phase 1 not run yet. Run Phase 1 first.")
                return None
        else:
            print(
                "Phase 1 intelligent core not trained yet. Train the Phase 1 intelligent core first."
            )
            logging.info(
                "Phase 1 intelligent core not trained yet. Train the Phase 1 intelligent core first."
            )
            return None

    # def run_phase2(self, phase1_object):
    #     """
    #     Creates an instance of IAOMAPhase2 using Phase 1 results.
    #     """
    #     if hasattr(self, "phase1"):  # if phase 1 already run
    #         print("Starting Phase 2...")
    #         logging.info("Starting Phase 2...")
    #         self.phase2 = IAOMAPhase2(phase1_object)
    #         self.phase2.loop_phase2_operations()
    #         # return phase2  # Return phase2 object for further use
    #     else:
    #         print("Phase 1 not run yet. Run Phase 1 first.")
    #         logging.info("Phase 1 not run yet. Run Phase 1 first.")
    #         return None

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
