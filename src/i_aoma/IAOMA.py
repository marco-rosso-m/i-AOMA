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
    def rf_intelligent_core_training(self, ICTHRESH: float = 0.2, output_path: str = os.getcwd()):
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
        
        self.dump_clf_model_to_file(output_path=self.output_path)

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


# ===================================================================================================== 
# SAVE RESULTS TO FILES
    def export_all_results_to_file(self, optional_prefix : str = ""):
        self.dump_sim_results_to_file(self.output_path + os.sep + optional_prefix)
        self.dump_clusters_to_file(self.output_path + os.sep + optional_prefix)

        print("All results exported to files.")
        logging.info("All results exported to files.")

    def dump_clf_model_to_file(self, output_path: str = os.getcwd()):
        with open(
            output_path
            + os.sep
            + f"clf_model.pkl",
            "wb",
        ) as backup_file:
            pickle.dump(self.clf_model, backup_file)

    def dump_sim_results_to_file(self, output_path: str = os.getcwd()):
        data_to_save = {
            'sim_results': self.Results.sim_results,
            'discarded_qmc_samples': self.Results.discarded_qmc_samples
        }
        with open(
            output_path
            + os.sep
            + f"sim_results_{len(self.Results.sim_results):d}_simulations_and_discarded_samples.pkl",
            "wb",
        ) as backup_file:
            pickle.dump(data_to_save, backup_file)

    def dump_clusters_to_file(self, output_path: str = os.getcwd()):
        data_to_save = {
            'clusters': self.Results.clusters,
            'clusters_id': self.Results.clusters_id,
            'KDEPROMINENCE': self.Results.KDEPROMINENCE , 
            'bw': self.Results.bw ,
            'current_freq_cluster_id': self.Results.current_freq_cluster_id
        }
        with open(
            output_path
            + os.sep
            + f"modal_clusters_{len(self.Results.sim_results):d}_simulations.pkl",
            "wb",
        ) as backup_file:
            pickle.dump(data_to_save, backup_file)

# ===================================================================================================== 
# LOADING FROM FILES
    def load_all_results_from_files(self, files_path: list):
        """
        Load all results from the specified files.

        Parameters:
        files_path (list of str): A list containing 2 file paths. Each path should be a string
                                    representing the location of a file to be loaded. The list should
                                    have the following format:
                                    [
                                        'path_to_sim_results.pkl',  # Path to the first file
                                        'path_to_discarded_qmc_samples.pkl',  # Path to the second file
                                        'path_to_clusters.pkl',  # Path to the third file
                                        'path_to_clusters_id.pkl'   # Path to the fourth file
                                    ]

        Returns:
        tuple: A tuple containing the loaded results from the four files.

        Raises:
        FileNotFoundError: If any of the files do not exist.
        """
        if len(files_path) != 2 :
            print("Error: 2 files are needed to load all results.")
            print("""Please provide a list of the path for the following 2 files in this specific order: 
                  file 1 is a dict containing two keys sim_results and discarded_qmc_samples
                  file 1 is a dict containing two keys clusters and clusters_id """)
        else:
            try:
                error_flag = 0
                self.load_sim_results_from_file(files_path[0])
                error_flag = 1
                self.load_clusters_from_file(files_path[1])
            except Exception as e:
                if error_flag == 0:
                    print(f"Error loading sim_results file: {e}")
                elif error_flag == 1:
                    print(f"Error loading clusters file: {e}")



    def load_clf_model_from_file(self, file_path):
        try:
            with open(file_path, 'rb') as file:
                self.clf_model = pickle.load(file)
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    

    def load_sim_results_from_file(self, file_path):
        """        
            data_stored = {
                'sim_results': self.Results.sim_results,
                'discarded_qmc_samples': self.Results.discarded_qmc_samples
            }
        """
        try:
            with open(file_path, 'rb') as file:
                loaded_data = pickle.load(file)
            self.Results.sim_results = loaded_data['sim_results']
            self.Results.discarded_qmc_samples = loaded_data['discarded_qmc_samples']
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
        
    def load_clusters_from_file(self, file_path):
        """        
            data_to_save = {
                'clusters': self.Results.clusters,
                'clusters_id': self.Results.clusters_id,
                'KDEPROMINENCE': self.Results.KDEPROMINENCE , 
                'bw': self.Results.bw ,
                'current_freq_cluster_id': self.Results.current_freq_cluster_id
            }
        """
        try:
            with open(file_path, 'rb') as file:
                loaded_data = pickle.load(file)
            self.Results.clusters = loaded_data['clusters']
            self.Results.clusters_id = loaded_data['clusters_id']
            self.Results.KDEPROMINENCE = loaded_data['KDEPROMINENCE']
            self.Results.bw = loaded_data['bw']
            self.Results.current_freq_cluster_id = loaded_data['current_freq_cluster_id']
        except Exception as e:
            print(f"Error loading file: {e}")
            return None