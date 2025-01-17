import numpy as np
import logging
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import os
import pickle
import glob

from .helper_ssicov_timeout import run_SSICov_with_timeout, update_heatmap

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
        self.Nsim_batch = iaoma.Nsim_batch
        self.NsimPh_max = iaoma.NsimPh_max
        self.Results = iaoma.Results
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

        self.sim_results = []  # list of dict to store results of each simulation containing Fn_poles, Lab, Xi_poles, Phi_poles, qmc_sample, IC
        self.discarded_qmc_samples = []  # list of list to store discarded qmc samples, columns contain qmc_sample, qmc_sample_unitary

        self.plt_resolution = iaoma.plt_resolution
        self.clf_model = iaoma.clf_model

        self.output_path_phase = (
            f"{self.output_path}" + os.sep + f"{iaoma.running_phase}"
        )
        if not os.path.exists(self.output_path_phase):
            os.makedirs(self.output_path_phase)

        self.output_path_phase_progroverlap = (
            f"{self.output_path}"
            + os.sep
            + f"{iaoma.running_phase}"
            + os.sep
            + "Progressive_overlapped_stab_diag"
        )
        if not os.path.exists(self.output_path_phase_progroverlap):
            os.makedirs(self.output_path_phase_progroverlap)
        else:
            for file in glob.glob(self.output_path_phase_progroverlap + os.sep + "*"):
                os.remove(file)

        self.output_path_phase_stab_diag_backup = (
            f"{self.output_path}"
            + os.sep
            + f"{iaoma.running_phase}"
            + os.sep
            + "Stab_diag_every_analysis_backups"
        )
        if not os.path.exists(self.output_path_phase_stab_diag_backup):
            os.makedirs(self.output_path_phase_stab_diag_backup)

        if not os.path.exists(
            self.output_path_phase_stab_diag_backup + os.sep + "Stab_diag"
        ):
            os.makedirs(self.output_path_phase_stab_diag_backup + os.sep + "Stab_diag")
        else:
            for file in glob.glob(
                self.output_path_phase_stab_diag_backup
                + os.sep
                + "Stab_diag"
                + os.sep
                + "*"
            ):
                os.remove(file)

        if not os.path.exists(
            self.output_path_phase_stab_diag_backup + os.sep + "Damping_Freq_diag"
        ):
            os.makedirs(
                self.output_path_phase_stab_diag_backup + os.sep + "Damping_Freq_diag"
            )
        else:
            for file in glob.glob(
                self.output_path_phase_stab_diag_backup
                + os.sep
                + "Damping_Freq_diag"
                + os.sep
                + "*"
            ):
                os.remove(file)

    def loop_phase_operations(
        self,
        n_jobs: int = -1,
        timeout_seconds: int = 30,
        set_seed=None,
        plt_stab_diag_backup: bool = False,
        progressive_plot_flag: bool = True,
    ):
        """
        Loop operations until collecting NsimPh1 results.
        """
        self.set_seed = set_seed
        self.n_jobs = n_jobs

        # Sequential loop
        if n_jobs == 0:
            print("Running IAOMA-Phase 1 (sequential mode)...")
            logging.info(
                "====================================================================================================="
            )
            logging.info("Running IAOMA-Phase 1 (sequential mode)...")

            batch_checkpoints = list(
                range(self.Nsim_batch - 1, self.NsimPh_max + 1, self.Nsim_batch)
            )

            for sim in range(
                0, self.NsimPh_max
            ):  # it is like having a batch equal to 1
                simbatch_res = run_SSICov_with_timeout(
                    self.SingleSetup,
                    self.qmc_limits,
                    timeout_seconds,
                    sim,
                    set_seed=set_seed,
                    plt_stab_diag_backup=plt_stab_diag_backup,
                    output_path_stab_diag_backup=self.output_path_phase_stab_diag_backup,
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
                                plt_resolution=self.plt_resolution,
                                sim=sim,
                                update_flag=False,
                            )
                        )
                        plt.pause(0.001)
                        plt.savefig(
                            f"{self.output_path_phase_progroverlap}/Overlap_Stab_Diag_until_sim_{1000+sim+1:d}.png",
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
                                plt_resolution=self.plt_resolution,
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
                            f"{self.output_path_phase_progroverlap}/Overlap_Stab_Diag_until_sim_{1000+sim+1:d}.png",
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
            for sim in range(0, self.NsimPh_max, self.Nsim_batch):
                print(f"Running batch {sim} to {sim+self.Nsim_batch-1}...")
                logging.info(
                    "#########################################################"
                )
                logging.info(f"Running batch {sim} to {sim+self.Nsim_batch-1}...")

                simbatch_res = Parallel(n_jobs=n_jobs)(
                    delayed(run_SSICov_with_timeout)(
                        self.SingleSetup,
                        self.qmc_limits,
                        timeout_seconds,
                        Nanal,
                        set_seed,
                        plt_stab_diag_backup,
                        self.output_path_phase_stab_diag_backup,
                    )
                    for Nanal in range(sim, sim + self.Nsim_batch)
                )

                for ii in range(self.Nsim_batch):
                    self.sim_results.append(
                        simbatch_res[ii][0]
                    )  # sim_results[ID] -> every ID element contains dict_keys(['Fn_poles', 'Xi_poles', 'Phi_poles', 'qmc_sample', 'IC'])
                    self.discarded_qmc_samples.extend(
                        simbatch_res[ii][1]
                    )  # discarded_qmc_samples[ID] -> every ID element contains two lists [[qmc_sample, qmc_sample_unitary]]

                # plot overlapped stab diag density every batch of analysis, diagnostic plot
                if sim == 0 and progressive_plot_flag:
                    # plot_overlap_stab_diag()

                    fig, ax, heatmap, xedges, yedges, im = (
                        self._progressive_plot_overlap_stab_diag(
                            self.sim_results,
                            plt_resolution=self.plt_resolution,
                            sim=self.Nsim_batch - 1,
                            update_flag=False,
                        )
                    )
                    plt.pause(0.001)
                    plt.savefig(
                        f"{self.output_path_phase_progroverlap}/Overlap_Stab_Diag_until_sim_{1000+self.Nsim_batch-1:d}.png",
                        dpi=200,
                    )

                    # TODO: implement a function to update the heatmap related to damping vs freq cluster
                elif progressive_plot_flag:
                    # update_overlap_stab_diag()
                    new_sim_results = []
                    for ii in range(sim, sim + self.Nsim_batch):
                        new_sim_results.append(self.sim_results[ii])
                    fig, ax, heatmap, xedges, yedges, im = (
                        self._progressive_plot_overlap_stab_diag(
                            new_sim_results,
                            plt_resolution=self.plt_resolution,
                            sim=sim + self.Nsim_batch - 1,
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
                        f"{self.output_path_phase_progroverlap}/Overlap_Stab_Diag_until_sim_{1000+sim+self.Nsim_batch-1:d}.png",
                        dpi=200,
                    )

                    # TODO: implement a function to update the heatmap related to damping vs freq cluster
                else:
                    fig, ax = None, None

        print("i-AOMA phase 1 analyses done!")
        logging.info("i-AOMA phase 1 analyses done!")
        logging.info(
            "====================================================================================================="
        )
        self.Results.add_new_sim_results(self.sim_results)
        self.Results.add_new_discarded_qmc_samples(self.discarded_qmc_samples)

        return fig, ax

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

    # OLD FUNCTIONS DA RIVEDERE

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

    # def _explore_predictions_of_trained_rf_NOT_WORKING(
    #     self, no_bins_resolution: int = 20, plot_admissible_predictions: bool = True
    # ):
    #     # NOTE: not complete function
    #     def generate_meshgrid(qmc_limits, num_points=100):
    #         ts_range = np.linspace(0.0, 1.0, num_points)
    #         order_range = np.linspace(0.0, 1.0, num_points)
    #         wlen_range = np.linspace(0.0, 1.0, num_points)
    #         tt_range = np.linspace(0.0, 1.0, num_points)

    #         ts, order, wlen, tt = np.meshgrid(
    #             ts_range, order_range, wlen_range, tt_range
    #         )
    #         return ts, order, wlen, tt

    #     # Generate meshgrid
    #     ts, order, wlen, tt = generate_meshgrid(self.qmc_limits, 20)

    #     # Predict using RF model
    #     mesh_points = np.c_[ts.ravel(), order.ravel(), wlen.ravel(), tt.ravel()]

    #     predictions = []
    #     for ii in range(0, mesh_points.shape[0], 1000):
    #         predictions.append(self.clf_model.predict(mesh_points[ii : ii + 1000, :]))
    #     predictions = np.concatenate(predictions)
    #     mesh_points_with_prediction = np.c_[
    #         ts.ravel(), order.ravel(), wlen.ravel(), tt.ravel(), predictions.ravel()
    #     ]

    #     # df = pd.DataFrame(mesh_points_with_prediction, columns=["Time Shift", "Order", "Window Length", "Time Target", "Prediction"])

    #     # sns.pairplot(df, hue="Prediction", plot_kws={"alpha": 0.1}, kind='kde')

    #     # sns.pairplot(df[df['Prediction']<1], hue="Prediction", plot_kws={"alpha": 0.5})
    #     # sns.pairplot(df[df['Prediction']>0], hue="Prediction", plot_kws={"alpha": 0.5})
    #     labels = ["Time shift", "Max Order", "Window Length", "Time Target"]

    #     num_vars = len(labels)
    #     fig, axes = plt.subplots(
    #         num_vars,
    #         num_vars,
    #         figsize=(9, 8),
    #         tight_layout=True,
    #         sharex="col",
    #         sharey="row",
    #     )

    #     # plot_title = fig.text(
    #     #     0.95,
    #     #     0.95,
    #     #     f"Explore predictions of RF (No. Sim. {len(self.sim_results)})",
    #     #     ha="right",
    #     #     va="center",
    #     #     fontsize=16,
    #     # )

    #     for ii in range(num_vars):
    #         for jj in range(num_vars):
    #             if ii > jj:
    #                 if plot_admissible_predictions:
    #                     # plot admissible predictions with label 1
    #                     x = mesh_points_with_prediction[
    #                         mesh_points_with_prediction[:, 4] > 0, jj
    #                     ]
    #                     y = mesh_points_with_prediction[
    #                         mesh_points_with_prediction[:, 4] > 0, ii
    #                     ]
    #                 else:
    #                     # plot the discarded predictions with label 0
    #                     x = mesh_points_with_prediction[
    #                         mesh_points_with_prediction[:, 4] < 1, jj
    #                     ]
    #                     y = mesh_points_with_prediction[
    #                         mesh_points_with_prediction[:, 4] < 1, ii
    #                     ]

    #                     if x.shape[0] == 0:
    #                         print("There are no discarded points predicted")
    #                         logging.error("There are no discarded points predicted")
    #                         # return None

    #                 # Compute the 2D histogram
    #                 heatmap, xedges, yedges = np.histogram2d(
    #                     x,
    #                     y,
    #                     bins=[no_bins_resolution, no_bins_resolution],
    #                     range=[[0.0, 1.0], [0.0, 1.0]],
    #                 )
    #                 heatmap = heatmap / np.max(
    #                     heatmap
    #                 )  # normalize the heatmap between 0 and 1

    #                 # Plot the heatmap
    #                 im = axes[ii, jj].imshow(
    #                     heatmap.T,
    #                     origin="lower",
    #                     aspect="auto",
    #                     extent=[0.0, 1.0, 0.0, 1.0],
    #                 )
    #                 if jj == 0:
    #                     axes[ii, jj].set_ylabel(labels[ii])
    #                 if ii == 3:
    #                     axes[ii, jj].set_xlabel(labels[jj])
    #                 if ii == 1 and jj == 0:
    #                     fig.colorbar(
    #                         im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04
    #                     )

    #                 # heatmaps_qmc_samples[f'{ii},{jj}'] = [im, heatmap, xedges, yedges]
    #                 # if i < 3 :
    #                 #     axes[i, j].sharex(axes[3, j])
    #                 #     axes[i, j].set_xticklabels([])
    #                 # if j > 1 :
    #                 #     axes[i, j].sharey(axes[i, 0])
    #                 #     axes[i, j].set_yticklabels([])
    #             else:
    #                 axes[ii, jj].axis("off")
