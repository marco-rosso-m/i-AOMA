import numpy as np
import logging
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import os
import pickle
import glob

from .helper_ssicov_timeout import run_SSICov_with_timeout, update_heatmap
from .IAOMAPhase1 import IAOMAPhase1

plt.ion()  # Enable interactive mode

class IAOMAPhase2(IAOMAPhase1):
    """
    Child class for Phase 2 operations.
    Inherits attributes from IAOMA and methods from IAOMAPhase1.
    Implements new functionality specific to Phase 2.
    """

    def __init__(self, iaoma):
        # Inherit attributes and methods from IAOMAPhase1 (and IAOMA indirectly)
        super().__init__(iaoma)

        self.convergence_threshold = iaoma.convergence_threshold
        self.convergence_reached = False

        self.output_path_phase_progr_results = (
            self.output_path_phase + os.sep + "progr_results_file_dump"
        )
        if not os.path.exists(self.output_path_phase_progr_results):
            os.makedirs(self.output_path_phase_progr_results)
        else:
            for file in glob.glob(
                self.output_path_phase_progr_results + os.sep + "*.pkl"
            ):
                os.remove(file)

        self.output_path_phase_progr_discarded_samples = (
            self.output_path_phase + os.sep + "progr_discarded_samples"
        )
        if not os.path.exists(self.output_path_phase_progr_discarded_samples):
            os.makedirs(self.output_path_phase_progr_discarded_samples)
        else:
            for file in glob.glob(
                self.output_path_phase_progr_discarded_samples + os.sep + "*.pkl"
            ):
                os.remove(file)

        # self.clf_model = clf_model

        # TODO: salvare i risultati della fase 1 in un file nella cartella phase2 come {1000+NsimPhase1}.pkl e caricarli qui
        # TODO: salvare i discarded qmc samples in una cartella discarded_qmc_samples_phase2

        # self.dump_sim_results_to_file(self.Results.sim_results)
        # self.dump_discarded_qmc_samples_to_file(self.Results.discarded_qmc_samples)

    # DUMP RESULTS TO FILES

    def _dump_sim_results_to_file(self, sim_results):
        with open(
            self.output_path_phase_progr_results
            + os.sep
            + f"{len(self.Results.sim_results):d}.pkl",
            "wb",
        ) as backup_file:
            pickle.dump(sim_results, backup_file)

    def _dump_discarded_qmc_samples_to_file(self, discarded_qmc_samples):
        for file in glob.glob(
            self.output_path_phase_progr_discarded_samples + os.sep + "*.pkl"
        ):
            os.remove(file)
        with open(
            self.output_path_phase_progr_discarded_samples
            + os.sep
            + "discarded_qmc_samples.pkl",
            "wb",
        ) as backup_file:
            pickle.dump(discarded_qmc_samples, backup_file)

    # def load_all_results_from_progressive_folders(self):
    #     results_list = []
    #     for file in glob.glob(self.output_path_phase_progr_results + os.sep + "*.pkl"):
    #         with open(file, "rb") as backup_file:
    #             results_list.extend(pickle.load(backup_file))

    #     with open(
    #         self.output_path_phase_progr_discarded_samples
    #         + os.sep
    #         + "discarded_qmc_samples.pkl",
    #         "rb",
    #     ) as backup_file:
    #         discarded_qmc_samples = pickle.load(backup_file)

    #     return results_list, discarded_qmc_samples

    # LOOP PHASE 2

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
        Nsim_phase1 = len(self.Results.sim_results)

        fig_rel_diff_traces = None

        if plt_stab_diag_backup:
            if not os.path.exists(self.output_path_phase_stab_diag_backup):
                os.makedirs(self.output_path_phase_stab_diag_backup)

        # Sequential loop
        if n_jobs == 0:
            print("Running IAOMA-Phase 2 (sequential mode)...")
            logging.info(
                "====================================================================================================="
            )
            logging.info("Running IAOMA-Phase 2 (sequential mode)...")

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
                    clf_model=self.clf_model,
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
            print("Running IAOMA-Phase 2 (parallel mode)...")
            logging.info(
                "====================================================================================================="
            )
            logging.info("Running IAOMA-Phase 2 (parallel mode)...")

            # TODO: Implement some controls on Nsim_batch and NsimPh1 to optimize the parallel loop
            for sim in range(
                Nsim_phase1, Nsim_phase1 + self.NsimPh_max, self.Nsim_batch
            ):
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
                        self.clf_model,
                    )
                    for Nanal in range(sim, sim + self.Nsim_batch)
                )

                self.sim_results = []
                self.discarded_qmc_samples = []

                for ii in range(self.Nsim_batch):
                    self.sim_results.append(
                        simbatch_res[ii][0]
                    )  # sim_results[ID] -> every ID element contains dict_keys(['Fn_poles', 'Xi_poles', 'Phi_poles', 'qmc_sample', 'IC'])
                    self.discarded_qmc_samples.extend(
                        simbatch_res[ii][1]
                    )  # discarded_qmc_samples[ID] -> every ID element contains two lists [[qmc_sample, qmc_sample_unitary]]

                self._dump_sim_results_to_file(self.sim_results)
                self._dump_discarded_qmc_samples_to_file(self.discarded_qmc_samples)

                self.Results.add_new_sim_results(self.sim_results)
                self.Results.add_new_discarded_qmc_samples(self.discarded_qmc_samples)

                _, _ = self.Results.normalized_kde_frequency_filtering(
                    KDEPROMINENCE=self.Results.KDEPROMINENCE
                )
                _ = self.Results.clusters_updates()

                # check convergence

                rel_diff_trace_along_sim = self.compute_rel_diff_trace_along_sim()

                if fig_rel_diff_traces is None:
                    fig_rel_diff_traces = (
                        self.plot_rel_difference_trace_cov_matrix_history(
                            rel_diff_trace_along_sim
                        )
                    )
                else:
                    fig_rel_diff_traces = (
                        self.plot_rel_difference_trace_cov_matrix_history(
                            rel_diff_trace_along_sim, fig=fig_rel_diff_traces
                        )
                    )

                for ii in range(self.Nsim_batch, len(self.Results.sim_results)):
                    traces_for_all_clusters = []
                    for jj, freq_id in enumerate(self.Results.clusters.keys()):
                        traces_for_all_clusters.append(
                            rel_diff_trace_along_sim[freq_id][ii - self.Nsim_batch : ii]
                            < self.convergence_threshold
                        )
                    traces_for_all_clusters = np.array(traces_for_all_clusters)
                    if np.all(traces_for_all_clusters):
                        logging.info(
                            "====================================================================================================="
                        )
                        print(
                            f"Mode shape convergence reached between simulations {ii-self.Nsim_batch:d}-{ii:d}!"
                        )
                        logging.info(
                            f"Mode shape convergence reached between simulations {ii-self.Nsim_batch:d}-{ii:d}!"
                        )
                        self.convergence_reached = True
                        break

                # plot overlapped stab diag density every batch of analysis, diagnostic plot
                if sim == Nsim_phase1 and progressive_plot_flag:
                    # plot_overlap_stab_diag()

                    fig, ax, heatmap, xedges, yedges, im = (
                        self._progressive_plot_overlap_stab_diag(
                            self.Results.sim_results,
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
                    # new_sim_results = []
                    # for ii in range(sim, sim + self.Nsim_batch):
                    #     new_sim_results.append(self.sim_results)
                    fig, ax, heatmap, xedges, yedges, im = (
                        self._progressive_plot_overlap_stab_diag(
                            self.sim_results,
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

                if self.convergence_reached:
                    break

        print("i-AOMA phase 2 analyses done!")
        logging.info("i-AOMA phase 2 analyses done!")
        logging.info(
            "====================================================================================================="
        )

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

    def tracking_results_along_simulations(self):
        results_over_clusters = {key: [] for key in self.Results.clusters.keys()}
        for ii, freq_id in enumerate(self.Results.clusters.keys()):
            poles_tracking_mean_freq = []
            poles_tracking_std_freq = []
            poles_tracking_mean_damp = []
            poles_tracking_std_damp = []
            poles_tracking_mean_mode_shapes = []
            poles_tracking_std_mode_shapes = []

            trace_cov_matrix = []
            for jj in range(len(self.Results.sim_results)):
                poles_tracking_mean_freq.append(
                    np.mean(
                        self.Results.sim_results[jj]["Fn_poles"][
                            self.Results.clusters[freq_id][jj]
                        ]
                    )
                )
                poles_tracking_std_freq.append(
                    np.std(
                        self.Results.sim_results[jj]["Fn_poles"][
                            self.Results.clusters[freq_id][jj]
                        ]
                    )
                )

                poles_tracking_mean_damp.append(
                    np.mean(
                        self.Results.sim_results[jj]["Xi_poles"][
                            self.Results.clusters[freq_id][jj]
                        ]
                    )
                )
                poles_tracking_std_damp.append(
                    np.std(
                        self.Results.sim_results[jj]["Xi_poles"][
                            self.Results.clusters[freq_id][jj]
                        ]
                    )
                )

                poles_tracking_mean_mode_shapes.append(
                    np.mean(
                        self.Results.sim_results[jj]["Phi_poles"][
                            self.Results.clusters[freq_id][jj]
                        ],
                        axis=0,
                    )
                )
                poles_tracking_std_mode_shapes.append(
                    np.std(
                        self.Results.sim_results[jj]["Phi_poles"][
                            self.Results.clusters[freq_id][jj]
                        ],
                        axis=0,
                    )
                )

                trace_cov_matrix.append(
                    np.trace(
                        np.cov(
                            self.Results.sim_results[jj]["Phi_poles"][
                                self.Results.clusters[freq_id][jj]
                            ].T
                        ).real
                    )
                )  # Example data: each row is a variable, each column is an observation

            results_over_clusters[freq_id] = {
                "mean_freq": poles_tracking_mean_freq,
                "std_freq": poles_tracking_std_freq,
                "mean_damp": poles_tracking_mean_damp,
                "std_damp": poles_tracking_std_damp,
                "mean_mode_shapes": poles_tracking_mean_mode_shapes,
                "std_mode_shapes": poles_tracking_std_mode_shapes,
                "trace_cov_matrix": trace_cov_matrix,
            }
        return results_over_clusters

    def progressive_tracking_along_sim(self, selected_var: str = "freq"):
        """selected_var = 'freq' or 'damp'"""
        if selected_var == "freq":
            selector_1 = "mean_freq"
            selector_2 = "std_freq"
        else:
            selector_1 = "mean_damp"
            selector_2 = "std_damp"

        results_over_clusters = self.tracking_results_along_simulations()

        fig = plt.figure(tight_layout=True)

        for ii, freq_id in enumerate(self.Results.clusters.keys()):
            plt.plot(
                np.arange(len(self.Results.sim_results)),
                results_over_clusters[freq_id][selector_1],
                label=f"{ii:d} - {freq_id:.2f} Hz",
            )
            plt.fill_between(
                np.arange(len(self.Results.sim_results)),
                np.array(results_over_clusters[freq_id][selector_1])
                - 3 * np.array(results_over_clusters[freq_id][selector_2]),
                np.array(results_over_clusters[freq_id][selector_1])
                + 3 * np.array(results_over_clusters[freq_id][selector_2]),
                alpha=0.2,
            )
        plt.legend()
        plt.xlabel("Simulation ID")
        if selected_var == "freq":
            plt.ylabel("Natural Frequency [Hz]")
        else:
            plt.ylabel("Damping Ratio [-]")

        return fig

    def progressive_mode_shape_component_tracking_along_sim(self, freq_id: float):
        results_over_clusters = self.tracking_results_along_simulations()

        fig = plt.figure(tight_layout=True)

        for jj in range(self.NDOFS):
            node_along_sim = []
            std_node_along_sim = []
            for kk in range(len(self.Results.sim_results)):
                node_along_sim.append(
                    results_over_clusters[freq_id]["mean_mode_shapes"][kk][jj]
                )
                std_node_along_sim.append(
                    results_over_clusters[freq_id]["std_mode_shapes"][kk][jj]
                )
            plt.plot(
                np.arange(len(self.Results.sim_results)),
                node_along_sim,
                "o-",
                label=f"{freq_id:.2f} Hz - Node {jj:d}",
                color=f"C{jj:d}",
            )
            plt.fill_between(
                np.arange(len(self.Results.sim_results)),
                np.array(node_along_sim) - 3 * np.array(std_node_along_sim),
                np.array(node_along_sim) + 3 * np.array(std_node_along_sim),
                alpha=0.2,
            )
            # plt.fill_between(np.arange(len(self.Results.sim_results)), np.array(results_over_clusters[freq_id][selector_1])-3*np.array(results_over_clusters[freq_id][selector_2]), np.array(results_over_clusters[freq_id][selector_1])+3*np.array(results_over_clusters[freq_id][selector_2]), alpha=0.2)
        plt.legend()
        plt.xlabel("Simulation ID")
        plt.ylabel("Mode shape component")

        return fig

    def compute_relative_difference_trace_cov_matrix_along_simulation(self, arr):
        relative_diff = [np.nan]
        for ii in range(1, len(arr)):
            diff = (arr[ii] - arr[ii - 1]) / arr[ii - 1]
            relative_diff.append(diff)
        return np.array(relative_diff)

    def compute_rel_diff_trace_along_sim(self):
        results_over_clusters = {key: [] for key in self.Results.clusters.keys()}

        for ii, freq_id in enumerate(self.Results.clusters.keys()):
            trace_cov_matrix = []
            for jj in range(1, len(self.Results.sim_results)):
                mode_shape_until_sim = []
                for kk in range(jj + 1):
                    mode_shape_until_sim.append(
                        self.Results.sim_results[kk]["Phi_poles"][
                            self.Results.clusters[freq_id][kk]
                        ]
                    )
                mode_shape_until_sim = np.concatenate(mode_shape_until_sim, axis=0)

                trace_cov_matrix.append(np.trace(np.cov(mode_shape_until_sim.T).real))

            results_over_clusters[freq_id] = trace_cov_matrix

        # aggrega risultati simulazione per simulazione
        rel_diff_trace_along_sim = {key: [] for key in self.Results.clusters.keys()}
        for ii, freq_id in enumerate(self.Results.clusters.keys()):
            rel_diff_trace_along_sim[freq_id] = (
                self.compute_relative_difference_trace_cov_matrix_along_simulation(
                    results_over_clusters[freq_id]
                )
            )

        return rel_diff_trace_along_sim

    def plot_rel_difference_trace_cov_matrix_history(
        self, rel_diff_along_sim, fig=None
    ):
        if fig is None:
            fig = plt.figure(tight_layout=True)
        else:
            fig.clf()  # clear the figure

        for ii, freq_id in enumerate(self.Results.clusters.keys()):
            plt.plot(
                np.arange(1, len(self.Results.sim_results)),
                rel_diff_along_sim[freq_id],
                ".-",
                color=f"C{ii:d}",
                label=f"{ii:d} - {freq_id:.2f} Hz",
            )

        plt.plot(
            np.arange(len(self.Results.sim_results)),
            self.convergence_threshold * np.ones(len(self.Results.sim_results)),
            "--",
            color="black",
            label="Convergence Threshold",
        )
        plt.plot(
            np.arange(len(self.Results.sim_results)),
            -self.convergence_threshold * np.ones(len(self.Results.sim_results)),
            "--",
            color="black",
        )
        plt.legend()
        plt.title("Rel. Diff. Trace of Total Sample Cov. Matrix")
        plt.xlabel("Simulation ID")
        plt.ylabel("Rel. Diff. Trace Cov. Matrix [-]")
        return fig
