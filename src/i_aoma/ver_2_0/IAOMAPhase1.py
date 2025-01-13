import numpy as np
import logging
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import os
from KDEpy import FFTKDE
from scipy.stats import beta
from scipy.signal import find_peaks
import pickle
from sklearn.ensemble import RandomForestClassifier

from .helper import run_SSICov_with_timeout, update_heatmap

# Set up logging
logging.basicConfig(
    filename="iaoma_run.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode="a",
)  # Use 'w' for overwrite mode, 'a' for append mode


class IAOMAPhase1:
    """
    Child class for Phase 1 operations.
    Inherits attributes from IAOMA but not its methods.
    List of methods:
    - loop_phase1_operations
    - _progressive_plot_overlap_stab_diag
    - plot_overlap_stab_diag
    - plot_overlap_freq_damp_cluster
    - normalize_kde_frequency_filtering
    - kde_clusters_selection
    - _kde_clusters_selection
    - _kde_dampling_selection_for_a_single_cluster
    - compute_ic_phase1
    - _compute_IC_for_new_sim
    - plot_ic_graph_over_all_sim
    """

    def __init__(self, iaoma):
        # Inherit attributes from IAOMA
        self.output_path = iaoma.output_path
        self.SingleSetup = iaoma.SingleSetup
        self.NDOFS = iaoma.NDOFS
        self.Ndata = iaoma.Ndata
        self.brmin = iaoma.brmin
        self.brmax = iaoma.brmax
        self.wlenmin = iaoma.wlenmin
        self.wlenmax = iaoma.wlenmax
        self.ordmin = iaoma.ordmin
        self.ordmax = iaoma.ordmax

        self.qmc_limits = {
            "brmin": self.brmin,
            "brmax": self.brmax,
            "ordmin": self.ordmin,
            "ordmax": self.ordmax,
            "wlenmin": self.wlenmin,
            "wlenmax": self.wlenmax,
            "NDOFS": self.NDOFS,
            "Ndata": self.Ndata,
        }

    def loop_phase1_operations(
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
        Loop operations until collecting NsimPh1 results.
        """
        self.NsimPh1 = NsimPh1
        self.Nsim_batch = Nsim_batch
        self.set_seed = set_seed
        self.n_jobs = n_jobs

        self.sim_results = []  # list of dict to store results of each simulation containing Fn_poles, Lab, Xi_poles, Phi_poles, qmc_sample, IC
        self.discarded_qmc_samples = []  # list of list to store discarded qmc samples, columns contain qmc_sample, qmc_sample_unitary

        self.output_path_phase1 = f"{self.output_path}/Phase1"
        if not os.path.exists(self.output_path_phase1):
            os.makedirs(self.output_path_phase1)

        self.output_path_phase1_progroverlap = (
            f"{self.output_path}/Phase1/Progressive_overlapped_stab_diag"
        )
        if not os.path.exists(self.output_path_phase1_progroverlap):
            os.makedirs(self.output_path_phase1_progroverlap)

        self.output_path_phase1_stab_diag_backup = (
            f"{self.output_path}/Phase1/Stab_diag_backups"
        )
        if plt_stab_diag_backup:
            if not os.path.exists(self.output_path_phase1_stab_diag_backup):
                os.makedirs(self.output_path_phase1_stab_diag_backup)

        # Sequential loop
        if n_jobs == 0:
            print("Running IAOMA-Phase 1 (sequential mode)...")
            logging.info(
                "====================================================================================================="
            )
            logging.info("Running IAOMA-Phase 1 (sequential mode)...")

            batch_checkpoints = list(range(Nsim_batch - 1, NsimPh1 + 1, Nsim_batch))

            for sim in range(0, NsimPh1):  # it is like having a batch equal to 1
                simbatch_res = run_SSICov_with_timeout(
                    self.SingleSetup,
                    self.qmc_limits,
                    timeout_seconds,
                    sim,
                    IC_pred=np.nan,
                    set_seed=set_seed,
                    plt_stab_diag_backup=plt_stab_diag_backup,
                    output_path_phase1_stab_diag_backup=self.output_path_phase1_stab_diag_backup,
                )
                self.sim_results.append(
                    simbatch_res[0]
                )  # sim_results[ID] -> every ID element contains dict_keys(['Fn_poles', 'Xi_poles', 'Phi_poles', 'qmc_sample', 'IC'])
                self.discarded_qmc_samples.extend(
                    simbatch_res[1]
                )  # discarded_qmc_samples[ID] -> every ID element contains two lists [[qmc_sample, qmc_sample_unitary]]

                # plot overlapped stab diag density every batch of analysis, diagnostic plot
                if sim in batch_checkpoints:
                    if sim == batch_checkpoints[0]:
                        # plot_overlap_stab_diag()

                        fig, ax, heatmap, xedges, yedges, im = (
                            self._progressive_plot_overlap_stab_diag(
                                self.sim_results,
                                plt_resolution=plt_resolution,
                                sim=sim,
                                update_flag=False,
                            )
                        )
                        plt.pause(0.001)
                        plt.savefig(
                            f"{self.output_path_phase1_progroverlap}/Overlap_Stab_Diag_until_sim_{1000+sim+1:d}.png",
                            dpi=200,
                        )

                        # TODO: implement a function to update the heatmap related to damping vs freq cluster
                    else:
                        # update_overlap_stab_diag()
                        new_sim_results = []
                        for ii in range(
                            batch_checkpoints[batch_checkpoints.index(sim) - 1] + 1,
                            batch_checkpoints[batch_checkpoints.index(sim)] + 1,
                        ):
                            new_sim_results.append(self.sim_results[ii])
                        fig, ax, heatmap, xedges, yedges, im = (
                            self._progressive_plot_overlap_stab_diag(
                                new_sim_results,
                                plt_resolution=plt_resolution,
                                sim=sim,
                                update_flag=True,
                                im=im,
                                heatmap=heatmap,
                                xedges=xedges,
                                yedges=yedges,
                                fig=fig,
                                ax=ax,
                            )
                        )
                        plt.pause(0.001)
                        plt.savefig(
                            f"{self.output_path_phase1_progroverlap}/Overlap_Stab_Diag_until_sim_{1000+sim+1:d}.png",
                            dpi=200,
                        )

                        # TODO: implement a function to update the heatmap related to damping vs freq cluster

        else:  # Parallel loop
            print("Running IAOMA-Phase 1 (parallel mode)...")
            logging.info(
                "====================================================================================================="
            )
            logging.info("Running IAOMA-Phase 1 (parallel mode)...")

            # TODO: Implement some controls on Nsim_batch and NsimPh1 to optimize the parallel loop
            for sim in range(0, NsimPh1, Nsim_batch):
                print(f"Running batch {sim} to {sim+Nsim_batch-1}...")
                logging.info(
                    "#########################################################"
                )
                logging.info(f"Running batch {sim} to {sim+Nsim_batch-1}...")

                simbatch_res = Parallel(n_jobs=n_jobs)(
                    delayed(run_SSICov_with_timeout)(
                        self.SingleSetup,
                        self.qmc_limits,
                        timeout_seconds,
                        Nanal,
                        np.nan,
                        set_seed,
                        plt_stab_diag_backup,
                        self.output_path_phase1_stab_diag_backup,
                    )
                    for Nanal in range(sim, sim + Nsim_batch)
                )

                for ii in range(Nsim_batch):
                    self.sim_results.append(
                        simbatch_res[ii][0]
                    )  # sim_results[ID] -> every ID element contains dict_keys(['Fn_poles', 'Xi_poles', 'Phi_poles', 'qmc_sample', 'IC'])
                    self.discarded_qmc_samples.extend(
                        simbatch_res[ii][1]
                    )  # discarded_qmc_samples[ID] -> every ID element contains two lists [[qmc_sample, qmc_sample_unitary]]

                if sim == 0:
                    # plot_overlap_stab_diag()

                    fig, ax, heatmap, xedges, yedges, im = (
                        self._progressive_plot_overlap_stab_diag(
                            self.sim_results,
                            plt_resolution=plt_resolution,
                            sim=Nsim_batch - 1,
                            update_flag=False,
                        )
                    )
                    plt.pause(0.001)
                    plt.savefig(
                        f"{self.output_path_phase1_progroverlap}/Overlap_Stab_Diag_until_sim_{1000+Nsim_batch-1:d}.png",
                        dpi=200,
                    )

                    # TODO: implement a function to update the heatmap related to damping vs freq cluster
                else:
                    # update_overlap_stab_diag()
                    new_sim_results = []
                    for ii in range(sim, sim + Nsim_batch):
                        new_sim_results.append(self.sim_results[ii])
                    fig, ax, heatmap, xedges, yedges, im = (
                        self._progressive_plot_overlap_stab_diag(
                            new_sim_results,
                            plt_resolution=plt_resolution,
                            sim=sim + Nsim_batch - 1,
                            update_flag=True,
                            im=im,
                            heatmap=heatmap,
                            xedges=xedges,
                            yedges=yedges,
                            fig=fig,
                            ax=ax,
                        )
                    )
                    plt.pause(0.001)
                    plt.savefig(
                        f"{self.output_path_phase1_progroverlap}/Overlap_Stab_Diag_until_sim_{1000+sim+Nsim_batch-1:d}.png",
                        dpi=200,
                    )

                    # TODO: implement a function to update the heatmap related to damping vs freq cluster

        print("i-AOMA phase 1 analyses done!")
        logging.info("i-AOMA phase 1 analyses done!")
        logging.info(
            "====================================================================================================="
        )
        return fig, ax

    # TODO: Implement the following functions
    # def plot_single_stab_diag(self, sim_id: int):
    #     step = 1
    #     x = list(self.sim_results[sim_id]['Fn_poles'].flatten(order="f"))
    #     y = list(np.array([i // len(self.sim_results[sim_id]['Fn_poles']) for i in range(self.sim_results[sim_id]['Fn_poles'].flatten(order="f").shape[0])]) * step)
    #     fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
    #     ax.set_title(   f"Nsim {sim_id:d}, (br {self.sim_results[sim_id]['qmc_sample'][0][0]:.0f}, " + \
    #                     f"ord {self.sim_results[sim_id]['qmc_sample'][1][0]:.0f}, " + \
    #                     f"wlen {self.sim_results[sim_id]['qmc_sample'][2][1]*100:.1f}$\%$, " + \
    #                     f"tt {self.sim_results[sim_id]['qmc_sample'][3][1]*100:.1f}$\%$)"    )
    #     ax.set_ylabel("Model Order [-]")
    #     ax.set_xlabel("Frequency [Hz]")
    #     ax.plot(x, y, "go", markersize=7, color = 'forestgreen')
    #     return fig, ax

    def _progressive_plot_overlap_stab_diag(
        self,
        list_sim_results: list,
        plt_resolution,
        sim=0,
        update_flag=False,
        im=None,
        heatmap=None,
        xedges=None,
        yedges=None,
        fig=None,
        ax=None,
    ):
        step = 1
        x = []
        y = []
        if not update_flag:
            for ii, res in enumerate(list_sim_results):
                x.extend(list(res["Fn_poles"].flatten(order="f")))
                y.extend(
                    list(
                        np.array(
                            [
                                i // len(res["Fn_poles"])
                                for i in range(
                                    res["Fn_poles"].flatten(order="f").shape[0]
                                )
                            ]
                        )
                        * step
                    )
                )
            data = np.array([x, y])
            data = data[:, ~np.isnan(data).any(axis=0)]
            fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
            ax.set_title(
                f"Overlap. Stab. Diag. Density Heatmap (i-AOMA Phase 1, Nsim={sim+1:d})"
            )
            ax.set_ylabel("Model Order [-]")
            ax.set_xlabel("Frequency [Hz]")
            heatmap, xedges, yedges = np.histogram2d(
                data[0, :],
                data[1, :],
                bins=[
                    round(self.SingleSetup.fs / 2 / plt_resolution["freq"]),
                    int(self.ordmax / plt_resolution["order"]),
                ],
                range=[[0, self.SingleSetup.fs / 2], [0, self.ordmax]],
            )
            heatmap = heatmap / np.max(heatmap)  # normalize the heatmap between 0 and 1
            im = ax.imshow(
                heatmap.T,
                origin="lower",
                cmap="viridis",
                extent=[0, self.SingleSetup.fs / 2, 0, self.ordmax],
                aspect="auto",
            )
            fig.colorbar(im, ax=ax)
        else:
            # Function to update heatmap
            # def update_heatmap(new_x, new_y, heatmap, im, xedges, yedges):
            #     # global heatmap
            #     new_heatmap, _, _ = np.histogram2d(
            #         new_x, new_y, bins=[xedges, yedges]
            #     )  # np.histogram2d(new_x, new_y)#, \
            #     # bins=[round(SingleSetup.fs/2/plt_resolution['freq']),int(new_x[1,:].max()/plt_resolution['order'])], \
            #     # range = [[0, SingleSetup.fs/2], [new_x[1,:].min(), new_x[1,:].max()]])
            #     new_heatmap = new_heatmap / np.max(
            #         new_heatmap
            #     )
            #     heatmap += new_heatmap
            #     heatmap = heatmap / np.max(
            #         heatmap
            #     )  # normalize the heatmap between 0 and 1
            #     im.set_data(heatmap.T)
            #     plt.draw()
            #     return im, heatmap

            for ii, res in enumerate(list_sim_results):
                x.extend(list(res["Fn_poles"].flatten(order="f")))
                y.extend(
                    list(
                        np.array(
                            [
                                i // len(res["Fn_poles"])
                                for i in range(
                                    res["Fn_poles"].flatten(order="f").shape[0]
                                )
                            ]
                        )
                        * step
                    )
                )
            new_data = np.array([x, y])
            new_data = new_data[:, ~np.isnan(new_data).any(axis=0)]
            im, heatmap = update_heatmap(
                new_data[0, :], new_data[1, :], heatmap, im, xedges, yedges
            )
            ax.set_title(
                f"Overlap. Stab. Diag. Density Heatmap (i-AOMA Phase 1, Nsim={sim+1:d})"
            )

        return fig, ax, heatmap, xedges, yedges, im

    def plot_overlap_stab_diag(
        self,
        method: str = "density",
        plt_resolution: dict = {"freq": 0.5, "damp": 0.001, "order": 1},
    ):
        """
        Plot Overlapped Stabilization Diagram
        x = Fn_poles
        y = Order

        Input:
        method: str = 'density' or 'scatter', default='density'
        """
        step = 1
        x = []
        y = []
        for ii, res in enumerate(self.sim_results):
            x.extend(list(res["Fn_poles"].flatten(order="f")))
            y.extend(
                list(
                    np.array(
                        [
                            i // len(res["Fn_poles"])
                            for i in range(res["Fn_poles"].flatten(order="f").shape[0])
                        ]
                    )
                    * step
                )
            )

        data = np.array([x, y])
        data = data[:, ~np.isnan(data).any(axis=0)]

        if method == "scatter":
            fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
            ax.set_title(
                f"Overlap. Stab. Diag. (i-AOMA Phase 1, Nsim={self.NsimPh1:d})"
            )
            ax.set_ylabel("Model Order [-]")
            ax.set_xlabel("Frequency [Hz]")
            ax.plot(data[0, :], data[1, :], "go", markersize=7, alpha=0.1)
            # mplcursors.cursor()
        elif method == "density":
            fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
            ax.set_title(
                f"Overlap. Stab. Diag. Density Heatmap (i-AOMA Phase 1, Nsim={self.NsimPh1:d})"
            )
            ax.set_ylabel("Model Order [-]")
            ax.set_xlabel("Frequency [Hz]")
            heatmap, xedges, yedges = np.histogram2d(
                data[0, :],
                data[1, :],
                bins=[
                    round(self.SingleSetup.fs / 2 / plt_resolution["freq"]),
                    int(self.ordmax / plt_resolution["order"]),
                ],
                range=[[0, self.SingleSetup.fs / 2], [0, self.ordmax]],
            )
            heatmap = heatmap / np.max(heatmap)  # normalize the heatmap between 0 and 1
            im = ax.imshow(
                heatmap.T,
                origin="lower",
                cmap="viridis",
                extent=[0, self.SingleSetup.fs / 2, 0, self.ordmax],
                aspect="auto",
            )
            fig.colorbar(im, ax=ax)
        print("Plotting Phase 1 results: overlapped stabilization diagram...")
        logging.info("Plotting Phase 1 results: overlapped stabilization diagram...")

        #     annot = ax.annotate("", xy=(0,0), xytext=(10,10),
        #                         textcoords="offset points",
        #                         bbox=dict(boxstyle="round", fc="w"),
        #                         arrowprops=dict(arrowstyle="->"))
        #     annot.set_visible(False)

        #     def update_annot(event):
        #         if event.inaxes == ax:
        #             x, y = event.xdata, event.ydata
        #             annot.xy = (x, y)
        #             text = f"(f={x:.2f} Hz, order={y:.0f})"
        #             annot.set_text(text)
        #             annot.set_visible(True)
        #             fig.canvas.draw_idle()

        #     def on_click(event):
        #         if event.inaxes == ax:
        #             update_annot(event)
        #         else:
        #             annot.set_visible(False)
        #             fig.canvas.draw_idle()

        #     fig.canvas.mpl_connect("button_press_event", on_click)

        return fig, ax

    def plot_overlap_freq_damp_cluster(
        self,
        method: str = "density",
        plt_resolution: dict = {"freq": 0.5, "damp": 0.001, "order": 1},
    ):
        """
        Plot Overlapped Damping Cluster Diagram
        x = Fn_poles
        y = Xi_poles

        Input:
        method: str = 'density' or 'scatter', default='density'
        """
        x = []
        y = []
        for ii, res in enumerate(self.sim_results):
            x.extend(list(res["Fn_poles"].flatten(order="f")))
            y.extend(list(res["Xi_poles"].flatten(order="f")))

        data = np.array([x, y])
        data = data[:, ~np.isnan(data).any(axis=0)]

        if method == "scatter":
            fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
            ax.set_title(
                f"Overlap. Damping Cluster (i-AOMA Phase 1, Nsim={self.NsimPh1:d})"
            )
            ax.set_ylabel("Damping Ratio [-]")
            ax.set_xlabel("Frequency [Hz]")
            ax.plot(data[0, :], data[1, :], "go", markersize=7, alpha=0.1)
            # mplcursors.cursor()
        elif method == "density":
            fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
            ax.set_title(
                f"Overlap. Damping Cluster Density Heatmap (i-AOMA Phase 1, Nsim={self.NsimPh1:d})"
            )
            ax.set_ylabel("Damping Ratio [-]")
            ax.set_xlabel("Frequency [Hz]")
            heatmap, xedges, yedges = np.histogram2d(
                data[0, :],
                data[1, :],
                bins=[
                    round(self.SingleSetup.fs / 2 / plt_resolution["freq"]),
                    int(
                        self.SingleSetup["SSIcov"].run_params.hc["xi_max"]
                        / plt_resolution["damp"]
                    ),
                ],
                range=[
                    [0, self.SingleSetup.fs / 2],
                    [0, self.SingleSetup["SSIcov"].run_params.hc["xi_max"]],
                ],
            )
            heatmap = heatmap / np.max(heatmap)  # normalize the heatmap between 0 and 1
            im = ax.imshow(
                heatmap.T,
                origin="lower",
                cmap="viridis",
                extent=[
                    0,
                    self.SingleSetup.fs / 2,
                    0,
                    self.SingleSetup["SSIcov"].run_params.hc["xi_max"],
                ],
                aspect="auto",
            )
            fig.colorbar(im, ax=ax)
            plt.pause(0.001)
        print("Plotting Phase 1 results: overlapped damping cluster diagram...")
        logging.info("Plotting Phase 1 results: overlapped damping cluster diagram...")

        return fig, ax

    def normalized_kde_frequency_filtering(
        self,
        KDEPROMINENCE: float = 0.0,  # 0. means automatic, otherwise a user_defined float value is used
        beta_distribution_percentile: float = 0.99,  # percentile to be used to define the prominence threshold
        plt_resolution: dict = {"freq": 0.05, "damp": 0.001, "order": 1},
        fr_diff_threshold: float = 0.10,  # Determine precision if two frequencies can be considered different cluster based on a threshold
        signifiant_digits_cluster_fr_keys: int = 2,  # Determine the number of significant digits to round the frequency cluster key
    ):
        # NOTE: for now, the prominence is computed only in phase 1, and then used also in phase 2, what is changing is the bw in phase 2
        self.KDEPROMINENCE = KDEPROMINENCE  # 0. means automatic, otherwise a user_defined float value is used
        if abs(self.KDEPROMINENCE) < np.finfo(float).eps:
            beta_plotting = True  # plot the beta distribution fitted on KDE data
        else:
            beta_plotting = False
        step = 1
        freq, order, damping, sim_id, rowid, colid = [], [], [], [], [], []
        for ii, res in enumerate(self.sim_results):
            freq.extend(list(res["Fn_poles"].flatten(order="f")))
            order.extend(
                list(
                    np.array(
                        [
                            i // len(res["Fn_poles"])
                            for i in range(res["Fn_poles"].flatten(order="f").shape[0])
                        ]
                    )
                    * step
                )
            )
            damping.extend(list(res["Xi_poles"].flatten(order="f")))
            sim_id.extend([ii] * len(res["Fn_poles"].flatten(order="f")))

            rows_indexes, columns_indexes = np.indices(res["Fn_poles"].shape)
            rowid.extend(rows_indexes.flatten(order="f"))
            colid.extend(columns_indexes.flatten(order="f"))

        data = np.array([freq, order, damping, sim_id, rowid, colid])
        data = data[
            :, ~np.isnan(data).any(axis=0)
        ]  # each row of data contains respecitvely freq, order, damping, sim_id, rowid, colid

        data = data[
            :, np.argsort(data[0, :])
        ]  # sort data by frequency in increasing order

        # KDE along frequency
        KDE = FFTKDE(kernel="gaussian", bw="ISJ").fit(data[0, :])
        self.bw = KDE.bw
        kde_fr_x, kde_fr_y = KDE.evaluate(
            int(self.SingleSetup.fs / 2 / plt_resolution["freq"])
        )
        kde_fr_ynorm = kde_fr_y / max(kde_fr_y)

        if abs(self.KDEPROMINENCE) < np.finfo(float).eps:
            # automatic prominence definition based on confidence interval of a beta distribution
            # fitted on the KDE data
            beta_fit = beta.fit(kde_fr_ynorm, floc=0, fscale=1.001)
            alpha_par, beta_par, a, bminusa = beta_fit
            self.KDEPROMINENCE = beta.ppf(
                beta_distribution_percentile, alpha_par, beta_par, loc=a, scale=bminusa
            )

        # find all peaks according to prominence threshold
        peaksFFTKDE, _ = find_peaks(kde_fr_ynorm, prominence=self.KDEPROMINENCE)

        # Check if clusters attribute already exists
        # if not hasattr(self, 'clusters'): # print("Attribute 'attribute_name' does not exist.")
        # this means that self.freq_cluster_id is the first time clusters to be defined
        # naming clusters according to KDE peaks and define them univocally by leveraging their rounded precision
        # the other cluster should be at least away from freq_cluster_id according to fr_diff_threshold and +-bw

        # check if peaks are too close to be gathered in the same cluster or not
        # Filter peaks based on the frequency difference threshold

        # filter peaks based on the frequency difference threshold, and keep the max ynorm value for those within frequency threshold
        filtered_peaks = []
        for ii in range(len(peaksFFTKDE)):
            if ii == 0:
                filtered_peaks.append(peaksFFTKDE[ii])
            else:
                if (kde_fr_x[peaksFFTKDE[ii]] - kde_fr_x[peaksFFTKDE[ii - 1]]) >= max(
                    fr_diff_threshold, self.bw
                ):
                    filtered_peaks.append(peaksFFTKDE[ii])
                else:
                    if kde_fr_ynorm[peaksFFTKDE[ii]] > kde_fr_ynorm[filtered_peaks[-1]]:
                        filtered_peaks[-1] = peaksFFTKDE[ii]

        peaksFFTKDE = filtered_peaks

        self.freq_cluster_id = np.round(
            np.round(kde_fr_x[filtered_peaks] / fr_diff_threshold) * fr_diff_threshold,
            max(
                len(str(fr_diff_threshold).strip().split(".")[1]),
                signifiant_digits_cluster_fr_keys,
            ),
        )

        # else:
        #     # check if the attribute exists then check if there are new clusters
        #     # or they are the same around of the firstly founded
        #     print("Attribute 'attribute_name' exists.")
        #     # TODO: check if the new clusters are the same as the previous ones or create a new label for the new cluster
        #     filtered_peaks = []
        #     for ii in range(len(peaksFFTKDE)):
        #         if ii == 0:
        #             filtered_peaks.append(peaksFFTKDE[ii])
        #         else:
        #             if (kde_fr_x[peaksFFTKDE[ii]] - kde_fr_x[peaksFFTKDE[ii-1]]) >= max(fr_diff_threshold, self.bw) :
        #                 filtered_peaks.append(peaksFFTKDE[ii])
        #             else:
        #                 if kde_fr_ynorm[peaksFFTKDE[ii]] > kde_fr_ynorm[filtered_peaks[-1]]:
        #                     filtered_peaks[-1] = peaksFFTKDE[ii]

        #     peaksFFTKDE = filtered_peaks

        #     freq_cluster_id_tmp = np.round(
        #         np.round(kde_fr_x[filtered_peaks] / fr_diff_threshold) * fr_diff_threshold,
        #         max(len(str(plt_resolution['freq']).strip().split('.')[1]), len(str(fr_diff_threshold).strip().split('.')[1]))
        #         )
        #     # check if tmp are the same of old cluster or not but it is done in the next step or notify the user to check the clusters
        if hasattr(self, "clusters"):
            if not np.array_equal(self.freq_cluster_id, list(self.clusters.keys())):
                print(
                    f"New clusters appeared after {len(self.sim_results):d} simulations."
                )
                logging.info(
                    f"New clusters appeared after {len(self.sim_results):d} simulations."
                )

        if beta_plotting:
            fig, ax = plt.subplots(
                1,
                2,
                figsize=(10, 4),
                tight_layout=True,
                gridspec_kw={"width_ratios": [1, 3]},
            )

            ax[0].set_title("Fitted Beta Distribution")
            ax[0].plot(
                beta.pdf(kde_fr_ynorm, alpha_par, beta_par, loc=a, scale=bminusa),
                kde_fr_ynorm,
                color="red",
                label="Beta Distribution",
            )
            ax[0].plot(
                [
                    0,
                    max(
                        beta.pdf(
                            kde_fr_ynorm, alpha_par, beta_par, loc=a, scale=bminusa
                        )
                    ),
                ],
                [self.KDEPROMINENCE, self.KDEPROMINENCE],
                "--",
                color="black",
                label="Auto. Promin. Thresh.",
            )
            ax[0].set_ylim(0, 1)
            ax[0].set_xlim(
                0,
                max(beta.pdf(kde_fr_ynorm, alpha_par, beta_par, loc=a, scale=bminusa)),
            )
            ax[0].invert_xaxis()
            ax[0].legend()

            ax[1].set_title(
                f"KDE along frequency (Total No Sim. {len(self.sim_results):d})"
            )
            ax[1].set_ylabel("Normalized KDE [-]")
            ax[1].set_xlabel("Frequency [Hz]")
            ax[1].plot(
                kde_fr_x, kde_fr_ynorm, color="blue", label=f"KDE (bw={self.bw:.2e})"
            )
            ax[1].plot(
                kde_fr_x[peaksFFTKDE],
                kde_fr_ynorm[peaksFFTKDE],
                "ro",
                markersize=7,
                label=f"Sel. Peaks (prom. > {self.KDEPROMINENCE:.2e})",
            )
            ax[1].legend()
            ax[1].set_ylim(0, 1)
            ax[1].set_xlim(0, self.SingleSetup.fs / 2)

        else:
            fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
            ax.set_title(
                f"KDE along frequency (Total No Sim. {len(self.sim_results):d})"
            )
            ax.set_ylabel("Normalized KDE [-]")
            ax.set_xlabel("Frequency [Hz]")
            ax.plot(
                kde_fr_x, kde_fr_ynorm, color="blue", label=f"KDE (bw={self.bw:.2e})"
            )
            ax.plot(
                kde_fr_x[peaksFFTKDE],
                kde_fr_ynorm[peaksFFTKDE],
                "ro",
                markersize=7,
                label=f"Sel. Peaks (prom. > {self.KDEPROMINENCE:.2e})",
            )
            ax.legend()
            plt.ylim(0, 1)
            plt.xlim(0, self.SingleSetup.fs / 2)

        # NOTE: Compute marginal distributions is almost equivalent to KDE,
        # but KDE is a non-parametric method that can estimate the nat freq
        # in a more precise way, rather than histogram2d which depends on the
        # frequency resolution (it could be used as a rapid estimate of prob distr on a grid)

        # marginal_x = np.sum(heatmap, axis=1)
        # marginal_x = marginal_x/max(marginal_x)
        # fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
        # ax.set_title(
        #     f"i-AOMA Phase 1: Marginal Distribution along frequency"
        # )
        # ax.set_ylabel("Normalized KDE [-]")
        # ax.set_xlabel("Frequency [Hz]")
        # ax.plot(xedges[:-1], marginal_x)
        # plt.ylim(0,1);plt.xlim(0,self.SingleSetup.fs/2)

        print(
            f"Computed KDE along natural frequency (No simulations so far {len(self.sim_results):d})."
        )
        logging.info(
            f"KDE natural frequency cluster (No simulations so far {len(self.sim_results):d})."
        )

        return fig, ax

    def kde_clusters_selection(
        self,
        plt_resolution: dict = {"freq": 0.05, "damp": 0.001, "order": 1},
        plot_clusters_damp_kde: bool = True,
    ):
        return self._kde_clusters_selection(
            self.sim_results,
            first_sim_no_of_list_sim_results=0,
            plt_resolution=plt_resolution,
            plot_clusters_damp_kde=plot_clusters_damp_kde,
        )

    def _kde_clusters_selection(
        self,
        list_sim_results: list,
        first_sim_no_of_list_sim_results: int = 0,
        plt_resolution: dict = {"freq": 0.05, "damp": 0.001, "order": 1},
        plot_clusters_damp_kde: bool = True,
    ):
        step = 1
        freq, order, damping, sim_id, rowid, colid = [], [], [], [], [], []
        for ii, res in enumerate(list_sim_results):
            freq.extend(list(res["Fn_poles"].flatten(order="f")))
            order.extend(
                list(
                    np.array(
                        [
                            i // len(res["Fn_poles"])
                            for i in range(res["Fn_poles"].flatten(order="f").shape[0])
                        ]
                    )
                    * step
                )
            )
            damping.extend(list(res["Xi_poles"].flatten(order="f")))
            sim_id.extend(
                [first_sim_no_of_list_sim_results + ii]
                * len(res["Fn_poles"].flatten(order="f"))
            )

            rows_indexes, columns_indexes = np.indices(res["Fn_poles"].shape)
            rowid.extend(rows_indexes.flatten(order="f"))
            colid.extend(columns_indexes.flatten(order="f"))

        data = np.array([freq, order, damping, sim_id, rowid, colid])
        data = data[
            :, ~np.isnan(data).any(axis=0)
        ]  # each row of data contains respecitvely freq, order, damping, sim_id, rowid, colid

        data = data[
            :, np.argsort(data[0, :])
        ]  # sort data by frequency in increasing order

        if not hasattr(
            self, "clusters"
        ):  #'attribute_name' does not exist , i.e. first time that clusters are defined          # # len(list(clusters.keys())) == 0:
            clusters = {}

            for freq_id in self.freq_cluster_id:
                _, _, cluster1 = self._kde_damping_selection_for_a_single_cluster(
                    data, freq_id, plt_resolution, plot_clusters_damp_kde
                )

                # construct masking boolean matrix to select poles within the list of sim_results to be associated to every cluster
                mask_list_for_each_cluster = []
                for ii, res in enumerate(list_sim_results):
                    bool_mask_for_sim = np.zeros(res["Fn_poles"].shape, dtype=bool)
                    idx_true = cluster1[
                        4:, (cluster1[3] == (first_sim_no_of_list_sim_results + ii))
                    ]  # boolean indexing mask
                    bool_mask_for_sim[
                        idx_true[0, :].astype(int), idx_true[1, :].astype(int)
                    ] = True
                    mask_list_for_each_cluster.append(bool_mask_for_sim)

                clusters[freq_id] = mask_list_for_each_cluster  # key are float values
                self.clusters = clusters  # store within the class

        else:  # if clusters are already defined, for every cluster the new boolean masks should be appended to the list of masks
            # TODO: if the cluster keys is not corresponding to the freq_cluster_id, then a new cluster should be added, with
            # all false masks for simulations before the current batch

            for freq_id in self.freq_cluster_id:
                if freq_id not in list(self.clusters.keys()):
                    print(
                        f"New cluster at {freq_id:.2f} Hz after {len(self.sim_results):d} simulations."
                    )
                    logging.info(
                        f"New cluster at {freq_id:.2f} Hz after {len(self.sim_results):d} simulations."
                    )

                    # add new cluster with all false masks for simulations before the current batch
                    _, _, cluster1 = self._kde_damping_selection_for_a_single_cluster(
                        data, freq_id, plt_resolution, plot_clusters_damp_kde
                    )

                    # construct masking boolean matrix to select poles within the list of sim_results to be associated to every cluster
                    mask_list_for_each_cluster = []
                    for ii in range(0, first_sim_no_of_list_sim_results):
                        bool_mask_for_sim = np.zeros(
                            list_sim_results[0]["Fn_poles"].shape, dtype=bool
                        )
                        mask_list_for_each_cluster.append(bool_mask_for_sim)
                    for ii, res in enumerate(list_sim_results):
                        bool_mask_for_sim = np.zeros(res["Fn_poles"].shape, dtype=bool)
                        idx_true = cluster1[
                            4:, (cluster1[3] == (first_sim_no_of_list_sim_results + ii))
                        ]  # boolean indexing mask
                        bool_mask_for_sim[
                            idx_true[0, :].astype(int), idx_true[1, :].astype(int)
                        ] = True
                        mask_list_for_each_cluster.append(bool_mask_for_sim)

                    self.clusters[freq_id] = (
                        mask_list_for_each_cluster  # store within the class
                    )

                else:  # cluster already exists, so add new masks to the list of masks for the specific cluster
                    _, _, cluster1 = self._kde_damping_selection_for_a_single_cluster(
                        data, freq_id, plt_resolution, plot_clusters_damp_kde
                    )

                    # construct masking boolean matrix to select poles within the list of sim_results to be associated to every cluster
                    mask_list_for_each_cluster = []
                    for ii, res in enumerate(list_sim_results):
                        bool_mask_for_sim = np.zeros(res["Fn_poles"].shape, dtype=bool)
                        idx_true = cluster1[
                            4:, (cluster1[3] == (first_sim_no_of_list_sim_results + ii))
                        ]  # boolean indexing mask
                        bool_mask_for_sim[
                            idx_true[0, :].astype(int), idx_true[1, :].astype(int)
                        ] = True
                        mask_list_for_each_cluster.append(bool_mask_for_sim)

                    self.clusters[freq_id].extend(
                        mask_list_for_each_cluster
                    )  # store within the class

        print(
            f"KDE natural frequency cluster selection completed (No simulations so far {len(self.sim_results):d})."
        )
        logging.info(
            f"KDE natural frequency cluster selection completed (No simulations so far {len(self.sim_results):d})."
        )

        return list(self.clusters.keys())  # return cluster id

    def _kde_damping_selection_for_a_single_cluster(
        self, data, freq_id, plt_resolution, plot_clusters_damp_kde
    ):
        cluster1 = data[
            :, (data[0, :] >= freq_id - self.bw) & (data[0, :] <= freq_id + self.bw)
        ]

        cluster1 = cluster1[
            :, np.argsort(cluster1[2, :])
        ]  # sort data by damping in increasing order within the cluster

        # KDE along frequency
        KDE_cluster = FFTKDE(kernel="gaussian", bw="ISJ").fit(cluster1[2, :])
        # bw_cluster = KDE_cluster.bw
        kde_xi_x, kde_xi_y = KDE_cluster.evaluate(
            int(
                self.SingleSetup["SSIcov"].run_params.hc["xi_max"]
                / plt_resolution["damp"]
            )
        )
        kde_xi_ynorm = kde_xi_y / max(kde_xi_y)

        # if KDEPROMINENCE_damp == 'automatic':
        #     # automatic prominence definition based on confidence interval of a beta distribution
        #     # fitted on the KDE data
        #     beta_fit_damp = beta.fit(kde_xi_ynorm, floc=0, fscale=1.001)
        #     alpha_par_damp, beta_par_damp, a_damp, bminusa_damp = beta_fit_damp
        #     KDEPROMINENCE_damp = beta.ppf(0.99, alpha_par_damp, beta_par_damp, loc=a_damp, scale=bminusa_damp)
        # # find peaks according to prominence threshold
        # peaksFFTKDE_damp, _ = find_peaks(kde_xi_ynorm, prominence=KDEPROMINENCE_damp)

        # NOTE: For now, the choice is taking in consideration only where the KDE reach value equal to 1
        # as the damping of interest, and taking all those poles within +- bw_cluster around the peak

        cluster1 = cluster1[
            :,
            (
                cluster1[2, :]
                >= kde_xi_x[np.argmax(kde_xi_ynorm)] - plt_resolution["damp"]
            )
            & (
                cluster1[2, :]
                <= kde_xi_x[np.argmax(kde_xi_ynorm)] + plt_resolution["damp"]
            ),
        ]
        if plot_clusters_damp_kde:
            fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
            ax.set_title(f"KDE along damping within cluster at {freq_id:.2f} Hz")
            ax.set_ylabel("Normalized KDE [-]")
            ax.set_xlabel("Damping Ratios [-]")
            ax.plot(
                kde_xi_x, kde_xi_ynorm, color="blue", label=f"KDE (bw={self.bw:.2e})"
            )
            ax.plot(
                kde_xi_x[np.argmax(kde_xi_ynorm)],
                kde_xi_ynorm[np.argmax(kde_xi_ynorm)],
                "ro",
                markersize=7,
                label="Abs. Max. Peak",
            )
            plt.ylim(0, 1)
            plt.xlim(0, self.SingleSetup["SSIcov"].run_params.hc["xi_max"])
            plt.plot(
                [kde_xi_x[np.argmax(kde_xi_ynorm)] - plt_resolution["damp"]] * 2,
                [0, 1],
                "r--",
                label="Retaining Bands",
            )
            plt.plot(
                [kde_xi_x[np.argmax(kde_xi_ynorm)] + plt_resolution["damp"]] * 2,
                [0, 1],
                "r--",
            )
            ax.legend()
            plt.ylim(0, 1)
            plt.xlim(0, self.SingleSetup["SSIcov"].run_params.hc["xi_max"])
            # mplcursors.cursor()
            return fig, ax, cluster1
        else:
            return None, None, cluster1

    def compute_ic_phase1(self):
        return self._compute_IC_for_new_sim(self.sim_results)

    def _compute_IC_for_new_sim(
        self, list_sim_results: list, first_sim_no_of_list_sim_results: int = 0
    ):
        IC_val_list = []
        # Compute IC for each simulation
        for ii, res in enumerate(list_sim_results):
            num_poles_within_retaining_bands_for_sim = 0
            for freq_id in list(self.clusters.keys()):
                num_poles_within_retaining_bands_for_sim += np.sum(
                    self.clusters[freq_id][first_sim_no_of_list_sim_results + ii]
                )
            num_stable_poles_for_sim = np.sum(
                ~np.isnan(res["Fn_poles"])
            )  # np.sum(~np.isnan(res['Fn_poles']))
            IC_computed = (
                num_poles_within_retaining_bands_for_sim / num_stable_poles_for_sim
            )
            list_sim_results[ii]["IC"][1] = IC_computed
            IC_val_list.append(IC_computed)
        return IC_val_list

    def plot_ic_graph(self):
        """
        Plot IC graph over all simulations
        """

        ic_values = []
        for ii, res in enumerate(self.sim_results):
            ic_values.append(res["IC"][1])

        fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
        ax.set_title("IC values")
        ax.set_ylabel("IC [-]")
        ax.set_xlabel("Simulation ID [-]")
        ax.plot(range(0, len(self.sim_results)), ic_values, "go")
        plt.ylim(0, 1)
        plt.xlim(0, len(self.sim_results))
        ax.set_xticks(range(0, len(self.sim_results)))

        return fig, ax

    def visualize_mode_shape_from_clusters(self):
        mode_shapes_clusters = {key: [] for key in self.clusters.keys()}
        for ii, res in enumerate(self.sim_results):
            for ff, freq_id in enumerate(list(self.clusters.keys())):
                # pass
                mask = self.clusters[freq_id][ii]
                if np.sum(mask) > 0:
                    mode_shapes_clusters[freq_id].append(res["Phi_poles"][mask, :])

        for ff, freq_id in enumerate(list(self.clusters.keys())):
            mode_shapes_clusters[freq_id] = np.concatenate(
                mode_shapes_clusters[freq_id]
            )

        for ff, freq_id in enumerate(list(self.clusters.keys())):
            plt.figure()
            modo_medio = np.mean(mode_shapes_clusters[freq_id].real, axis=0)
            modo_std_dev = np.std(mode_shapes_clusters[freq_id].real, axis=0)
            plt.plot(
                np.arange(0, modo_medio.shape[0]),
                modo_medio,
                ".-",
                label=f"Freq. {freq_id:.2f} Hz",
            )
            plt.fill_between(
                np.arange(0, modo_medio.shape[0]),
                modo_medio - 3 * modo_std_dev,
                modo_medio + 3 * modo_std_dev,
                alpha=0.2,
                color="C1",
            )
            plt.legend()
            plt.title(f"Mode Shape Cluster at {freq_id:.2f} Hz")
            plt.xlabel("Mode Shape Index")
            plt.ylabel("Amplitude")
            plt.tight_layout()

        # NOTE: test checking masking: check the number of trues with np.sum(self.clusters[freq_id][ii]) , mask = self.clusters[freq_id][ii]
        # list_sim_results[0]['Fn_poles'][mask]
        # list_sim_results[0]['Xi_poles'][mask]
        # list_sim_results[0]['Phi_poles'][mask]

    def _dump_metadata_to_file_phase1(self, output_path):
        if hasattr(self, "freq_cluster_id"):
            attributes_to_store = {
                "NsimPh1": self.NsimPh1,
                "Nsim_batch": self.Nsim_batch,
                "discarded_qmc_samples": self.discarded_qmc_samples,
                "KDEPROMINENCE": self.KDEPROMINENCE,
                "bw": self.bw,
                "freq_cluster_id": self.freq_cluster_id,
            }
            # Store the dictionary in a pickle file
            with open(
                output_path + os.sep + "phase1_metadata.pkl", "wb"
            ) as backup_file:
                pickle.dump(attributes_to_store, backup_file)

            print("Dumping to file of metadata of phase 1 completed!")
            logging.error("Dumping to file of metadata of phase 1 completed!")
        elif hasattr(self, "discarded_qmc_samples"):
            attributes_to_store = {
                "NsimPh1": self.NsimPh1,
                "Nsim_batch": self.Nsim_batch,
                "discarded_qmc_samples": self.discarded_qmc_samples,
            }
            # Store the dictionary in a pickle file
            with open(
                output_path + os.sep + "phase1_metadata.pkl", "wb"
            ) as backup_file:
                pickle.dump(attributes_to_store, backup_file)
        else:
            print("Error in dumping metadata to file for phase 1.")
            logging.error("Error in dumping metadata to file for phase 1.")

    def _load_metadata_from_file_phase1(self, input_path):
        try:
            with open(input_path, "rb") as backup_file:
                loaded_attributes = pickle.load(backup_file)
            for key, value in loaded_attributes.items():
                setattr(self, key, value)
        except Exception as e:
            print(f"Error in loading metadata from file for phase 1. {e}")
            logging.error(
                "****************************************************************"
            )
            logging.error("Error in loading metadata from file for phase 1.")

    def _dump_results_to_file_phase1(self, output_path):
        if hasattr(self, "clusters"):
            attributes_to_store = {
                "sim_results": self.sim_results,
                "clusters": self.clusters,
            }
            # Store the dictionary in a pickle file
            with open(output_path + os.sep + "phase1_results.pkl", "wb") as backup_file:
                pickle.dump(attributes_to_store, backup_file)
        elif hasattr(self, "discarded_qmc_samples"):
            attributes_to_store = {
                "sim_results": self.sim_results,
            }
            # Store the dictionary in a pickle file
            with open(output_path + os.sep + "phase1_results.pkl", "wb") as backup_file:
                pickle.dump(attributes_to_store, backup_file)
        else:
            print("Error in dumping results to file for phase 1.")
            logging.error("Error in dumping results to file for phase 1.")

    def _load_results_from_file_phase1(self, input_path):
        try:
            with open(input_path, "rb") as backup_file:
                loaded_attributes = pickle.load(backup_file)
            for key, value in loaded_attributes.items():
                setattr(self, key, value)
        except Exception as e:
            print(f"Error in loading results from file for phase 1. {e}")
            logging.error(
                "****************************************************************"
            )
            logging.error("Error in loading results from file for phase 1.")

    def visualize_qmc_samples_distribution(self):
        return self._progressive_visualize_qmc_samples_distribution(self.sim_results)

    def _progressive_visualize_qmc_samples_distribution(
        self,
        list_sim_results: list,
        sim=0,
        no_bins_resolution: int = 20,  # defined only the first time the plot is generated
        # heatmaps_qmc_samples : dict = {}
    ):
        """
        Reminder:
            HaltonSamples[:, 0] corresponds to the timeshift parameter.
            HaltonSamples[:, 1] corresponds to the order parameter.
            HaltonSamples[:, 2] corresponds to the window_length parameter.
            HaltonSamples[:, 3] corresponds to the time_target_centering_window parameter.
        """

        admissible_qmc_samples = []
        for ii, res in enumerate(list_sim_results):
            admissible_qmc_samples.append(res["qmc_sample"][:, 1])  # unitary.

        # Convert the list to a numpy array for easier slicing
        data = np.array(admissible_qmc_samples)

        # Define the pairs of indices for the variables
        # pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        labels = ["Time shift", "Max Order", "Window Length", "Time Target"]

        num_vars = len(labels)

        if not hasattr(
            self, "heatmaps_qmc_samples"
        ):  # and len(heatmaps_qmc_samples.keys()) == 0:
            fig, axes = plt.subplots(
                num_vars,
                num_vars,
                figsize=(9, 8),
                tight_layout=True,
                sharex="col",
                sharey="row",
            )

            plot_title = fig.text(
                0.95,
                0.95,
                f"Admiss. Quasi-MC Samples (No. Sim. {sim+len(list_sim_results)})",
                ha="right",
                va="center",
                fontsize=16,
            )

            heatmaps_qmc_samples = {
                "fig": fig,
                "axes": axes,
                "plot_title": plot_title,
                "no_bins_resolution": no_bins_resolution,
            }
            # heatmaps_qmc, xedges_qmc, y_edges_qmc = [], [], []
            # Create 2D histograms for each pair
            for ii in range(num_vars):
                for jj in range(num_vars):
                    if ii == jj:
                        # Create histogram
                        hist, bin_edges = np.histogram(
                            data, bins=no_bins_resolution, range=[0.0, 1.0]
                        )
                        # # Normalize the histogram
                        hist_normalized = hist / np.max(hist)
                        axes[ii, jj].bar(
                            bin_edges[:-1],
                            hist_normalized,
                            width=np.diff(bin_edges),
                            edgecolor="black",
                            align="edge",
                        )
                        if jj == 0:
                            axes[ii, jj].set_ylabel(labels[ii])
                        if ii == 3:
                            axes[ii, jj].set_xlabel(labels[ii])
                        if ii == 0 and jj == 0:
                            axes[ii, jj].set_ylim([0.0, 1.0])
                            # axes[i, j].set_yticklabels([])
                        heatmaps_qmc_samples[f"{ii},{jj}"] = [hist_normalized]
                    elif ii > jj:
                        x = data[:, jj]
                        y = data[:, ii]

                        # Compute the 2D histogram
                        heatmap, xedges, yedges = np.histogram2d(
                            x,
                            y,
                            bins=[no_bins_resolution, no_bins_resolution],
                            range=[[0.0, 1.0], [0.0, 1.0]],
                        )
                        heatmap = heatmap / np.max(
                            heatmap
                        )  # normalize the heatmap between 0 and 1

                        # Plot the heatmap
                        im = axes[ii, jj].imshow(
                            heatmap.T,
                            origin="lower",
                            aspect="auto",
                            extent=[0.0, 1.0, 0.0, 1.0],
                        )
                        if jj == 0:
                            axes[ii, jj].set_ylabel(labels[ii])
                        if ii == 3:
                            axes[ii, jj].set_xlabel(labels[jj])
                        if ii == 1 and jj == 0:
                            fig.colorbar(
                                im,
                                ax=axes,
                                orientation="vertical",
                                fraction=0.02,
                                pad=0.04,
                            )

                        heatmaps_qmc_samples[f"{ii},{jj}"] = [
                            im,
                            heatmap,
                            xedges,
                            yedges,
                        ]
                        # if i < 3 :
                        #     axes[i, j].sharex(axes[3, j])
                        #     axes[i, j].set_xticklabels([])
                        # if j > 1 :
                        #     axes[i, j].sharey(axes[i, 0])
                        #     axes[i, j].set_yticklabels([])
                    else:
                        axes[ii, jj].axis("off")
            self.heatmaps_qmc_samples = heatmaps_qmc_samples

        else:
            # NOTE: if the user needs to change no bins resolution, it is necessary to delete the existing attribute heatmaps_qmc_samples ( del self.heatmaps_qmc_samples ) and run again the function with the new bins resolution
            # Remove old text
            self.heatmaps_qmc_samples["plot_title"].remove()

            # Add new text
            plot_title = self.heatmaps_qmc_samples["fig"].text(
                0.95,
                0.95,
                f"Admiss. Quasi-MC Samples (No. Sim. {sim+len(list_sim_results)})",
                ha="right",
                va="center",
                fontsize=16,
            )
            self.heatmaps_qmc_samples["plot_title"] = plot_title

            for ii in range(num_vars):
                for jj in range(num_vars):
                    if ii == jj:
                        # Create histogram
                        hist, bin_edges = np.histogram(
                            data,
                            bins=self.heatmaps_qmc_samples["no_bins_resolution"],
                            range=[0.0, 1.0],
                        )
                        # # Normalize the histogram
                        hist_normalized = hist / np.max(hist)
                        hist_normalized += (
                            hist_normalized + self.heatmaps_qmc_samples[f"{ii},{jj}"][0]
                        )
                        self.heatmaps_qmc_samples[f"{ii},{jj}"][0] = (
                            hist_normalized / np.max(hist_normalized)
                        )  # normalize the heatmap between 0 and 1
                        # Clear the old graph
                        self.heatmaps_qmc_samples["axes"][ii, jj].cla()

                        self.heatmaps_qmc_samples["axes"][ii, jj].bar(
                            bin_edges[:-1],
                            self.heatmaps_qmc_samples[f"{ii},{jj}"][0],
                            width=np.diff(bin_edges),
                            edgecolor="black",
                            align="edge",
                        )
                        if jj == 0:
                            self.heatmaps_qmc_samples["axes"][ii, jj].set_ylabel(
                                labels[ii]
                            )
                        if ii == 3:
                            self.heatmaps_qmc_samples["axes"][ii, jj].set_xlabel(
                                labels[ii]
                            )
                        if ii == 0 and jj == 0:
                            self.heatmaps_qmc_samples["axes"][ii, jj].set_ylim(
                                [0.0, 1.0]
                            )
                            # axes[i, j].set_yticklabels([])
                        # heatmaps_qmc_samples[f'{ii},{jj}'] = [hist_normalized]

                        # if jj == 0:
                        #     axes[ii, jj].set_ylabel(labels[ii])
                        # if ii == 3:
                        #     axes[ii, jj].set_xlabel(labels[ii])
                        # if ii == 0 and jj == 0 :
                        #     axes[ii, jj].set_ylim([0., 1.])

                    elif ii > jj:
                        x = data[:, jj]
                        y = data[:, ii]

                        (
                            self.heatmaps_qmc_samples[f"{ii},{jj}"][0],
                            self.heatmaps_qmc_samples[f"{ii},{jj}"][1],
                        ) = update_heatmap(
                            x,
                            y,
                            self.heatmaps_qmc_samples[f"{ii},{jj}"][1],
                            self.heatmaps_qmc_samples[f"{ii},{jj}"][0],
                            self.heatmaps_qmc_samples[f"{ii},{jj}"][2],
                            self.heatmaps_qmc_samples[f"{ii},{jj}"][3],
                        )
                        # NOTE: reminder: heatmaps_qmc_samples[f'{ii},{jj}'] = [im, heatmap, xedges, yedges]

        # return self.heatmaps_qmc_samples

    def rf_intelligent_core_training(self, ICTHRESH: float = 0.2):
        if len(self.discarded_qmc_samples) == 0:
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
        for ii, res in enumerate(self.sim_results):
            ic_values.append(res["IC"][1])
            db_qmc_samples.append(
                res["qmc_sample"][:, 1]
            )  # unitary qmc admissible samples
        for ii, sample in enumerate(self.discarded_qmc_samples):
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

    def _explore_predictions_of_trained_rf_NOT_WORKING(
        self, no_bins_resolution: int = 20, plot_admissible_predictions: bool = True
    ):
        # NOTE: not complete function
        def generate_meshgrid(qmc_limits, num_points=100):
            ts_range = np.linspace(0.0, 1.0, num_points)
            order_range = np.linspace(0.0, 1.0, num_points)
            wlen_range = np.linspace(0.0, 1.0, num_points)
            tt_range = np.linspace(0.0, 1.0, num_points)

            ts, order, wlen, tt = np.meshgrid(
                ts_range, order_range, wlen_range, tt_range
            )
            return ts, order, wlen, tt

        # Generate meshgrid
        ts, order, wlen, tt = generate_meshgrid(self.qmc_limits, 20)

        # Predict using RF model
        mesh_points = np.c_[ts.ravel(), order.ravel(), wlen.ravel(), tt.ravel()]

        predictions = []
        for ii in range(0, mesh_points.shape[0], 1000):
            predictions.append(self.clf_model.predict(mesh_points[ii : ii + 1000, :]))
        predictions = np.concatenate(predictions)
        mesh_points_with_prediction = np.c_[
            ts.ravel(), order.ravel(), wlen.ravel(), tt.ravel(), predictions.ravel()
        ]

        # df = pd.DataFrame(mesh_points_with_prediction, columns=["Time Shift", "Order", "Window Length", "Time Target", "Prediction"])

        # sns.pairplot(df, hue="Prediction", plot_kws={"alpha": 0.1}, kind='kde')

        # sns.pairplot(df[df['Prediction']<1], hue="Prediction", plot_kws={"alpha": 0.5})
        # sns.pairplot(df[df['Prediction']>0], hue="Prediction", plot_kws={"alpha": 0.5})
        labels = ["Time shift", "Max Order", "Window Length", "Time Target"]

        num_vars = len(labels)
        fig, axes = plt.subplots(
            num_vars,
            num_vars,
            figsize=(9, 8),
            tight_layout=True,
            sharex="col",
            sharey="row",
        )

        # plot_title = fig.text(
        #     0.95,
        #     0.95,
        #     f"Explore predictions of RF (No. Sim. {len(self.sim_results)})",
        #     ha="right",
        #     va="center",
        #     fontsize=16,
        # )

        for ii in range(num_vars):
            for jj in range(num_vars):
                if ii > jj:
                    if plot_admissible_predictions:
                        # plot admissible predictions with label 1
                        x = mesh_points_with_prediction[
                            mesh_points_with_prediction[:, 4] > 0, jj
                        ]
                        y = mesh_points_with_prediction[
                            mesh_points_with_prediction[:, 4] > 0, ii
                        ]
                    else:
                        # plot the discarded predictions with label 0
                        x = mesh_points_with_prediction[
                            mesh_points_with_prediction[:, 4] < 1, jj
                        ]
                        y = mesh_points_with_prediction[
                            mesh_points_with_prediction[:, 4] < 1, ii
                        ]

                        if x.shape[0] == 0:
                            print("There are no discarded points predicted")
                            logging.error("There are no discarded points predicted")
                            # return None

                    # Compute the 2D histogram
                    heatmap, xedges, yedges = np.histogram2d(
                        x,
                        y,
                        bins=[no_bins_resolution, no_bins_resolution],
                        range=[[0.0, 1.0], [0.0, 1.0]],
                    )
                    heatmap = heatmap / np.max(
                        heatmap
                    )  # normalize the heatmap between 0 and 1

                    # Plot the heatmap
                    im = axes[ii, jj].imshow(
                        heatmap.T,
                        origin="lower",
                        aspect="auto",
                        extent=[0.0, 1.0, 0.0, 1.0],
                    )
                    if jj == 0:
                        axes[ii, jj].set_ylabel(labels[ii])
                    if ii == 3:
                        axes[ii, jj].set_xlabel(labels[jj])
                    if ii == 1 and jj == 0:
                        fig.colorbar(
                            im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04
                        )

                    # heatmaps_qmc_samples[f'{ii},{jj}'] = [im, heatmap, xedges, yedges]
                    # if i < 3 :
                    #     axes[i, j].sharex(axes[3, j])
                    #     axes[i, j].set_xticklabels([])
                    # if j > 1 :
                    #     axes[i, j].sharey(axes[i, 0])
                    #     axes[i, j].set_yticklabels([])
                else:
                    axes[ii, jj].axis("off")
