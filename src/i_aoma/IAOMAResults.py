import numpy as np
import logging
import matplotlib.pyplot as plt
from KDEpy import FFTKDE
from scipy.stats import beta
from scipy.signal import find_peaks
import pickle
import os

from pyoma2.functions.gen import MAC


class IAOMAResults:
    """
    Class for storing and postprocessing the IAOMA results.
    """

    def __init__(self, iaoma):
        self.sim_results = []
        self.discarded_qmc_samples = []
        self.clusters = {}
        self.clusters_id = []

        self.SingleSetup = iaoma.SingleSetup
        self.plt_resolution = iaoma.plt_resolution
        self.ordmax = iaoma.ordmax

        self.KDEPROMINENCE = 0.0
        self.bw = 0.0
        self.current_freq_cluster_id = []

        self.modal_cluster_resolution = iaoma.modal_cluster_resolution

    # %% ADD NEW RESULTS
    def add_new_sim_results(self, new_sim_results):
        """
        Add new simulation results to the list of simulation results.
        """
        self.sim_results.extend(new_sim_results)

    def add_new_discarded_qmc_samples(self, new_discarded_qmc_samples):
        """
        Add new_discarded_qmc_samples to the list of discarded_qmc_samples.
        """
        self.discarded_qmc_samples.extend(new_discarded_qmc_samples)

    # %% PLOT OVERLAPPED STAB DIAG AND DAMPING DIAG

    def plot_overlap_stab_diag(
        self,
        method: str = "density",
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
                f"Overlap. Stab. Diag. (i-AOMA Phase 1, Nsim={len(self.sim_results):d})"
            )
            ax.set_ylabel("Model Order [-]")
            ax.set_xlabel("Frequency [Hz]")
            ax.plot(data[0, :], data[1, :], "go", markersize=7, alpha=0.1)
            # mplcursors.cursor()
        elif method == "density":
            fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
            ax.set_title(
                f"Overlap. Stab. Diag. Density Heatmap (i-AOMA Phase 1, Nsim={len(self.sim_results):d})"
            )
            ax.set_ylabel("Model Order [-]")
            ax.set_xlabel("Frequency [Hz]")
            heatmap, xedges, yedges = np.histogram2d(
                data[0, :],
                data[1, :],
                bins=[
                    round(self.SingleSetup.fs / 2 / self.plt_resolution["freq"]),
                    int(self.ordmax / self.plt_resolution["order"]),
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

        return fig, ax

    def plot_overlap_freq_damp_cluster(
        self,
        method: str = "density",
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
                f"Overlap. Damping Cluster (i-AOMA Phase 1, Nsim={len(self.sim_results):d})"
            )
            ax.set_ylabel("Damping Ratio [-]")
            ax.set_xlabel("Frequency [Hz]")
            ax.plot(data[0, :], data[1, :], "go", markersize=7, alpha=0.1)
            # mplcursors.cursor()
        elif method == "density":
            fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
            ax.set_title(
                f"Overlap. Damping Cluster Density Heatmap (i-AOMA Phase 1, Nsim={len(self.sim_results):d})"
            )
            ax.set_ylabel("Damping Ratio [-]")
            ax.set_xlabel("Frequency [Hz]")
            heatmap, xedges, yedges = np.histogram2d(
                data[0, :],
                data[1, :],
                bins=[
                    round(self.SingleSetup.fs / 2 / self.plt_resolution["freq"]),
                    int(
                        self.SingleSetup["SSIcov"].run_params.hc["xi_max"]
                        / self.plt_resolution["damp"]
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

    # %% KDE FILTERING ALONG FREQUENCY

    def get_aggregated_data_format(self):
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
        ]  # data contains 6 rows, which are respecitvely freq, order, damping, sim_id, rowid, colid
        return data

    def normalized_kde_frequency_filtering(
        self,
        KDEPROMINENCE: float = 0.0,  # 0. means automatic, otherwise a user_defined float value is used
        beta_distribution_percentile: float = 0.9999,  # percentile to be used to define the prominence threshold
    ):
        """
        Filters the normalized KDE frequency data to identify significant peaks based on prominence and frequency difference thresholds.

        Parameters:
        -----------
        KDEPROMINENCE : float, optional
            Initial prominence threshold for peak detection. Default is 0.3.
        fr_diff_threshold : float, optional
            Minimum frequency difference threshold to distinguish between peaks. Default is 0.01.
        beta_distribution_percentile : float, optional
            Percentile value for the beta distribution to dynamically adjust the prominence threshold. Default is 0.95.
        beta_plotting : bool, optional
            If True, plots the fitted beta distribution and KDE. Default is False.
        signifiant_digits_cluster_fr_keys : int, optional
            Number of significant digits for rounding frequency cluster keys. Default is 2.

        Returns:
        --------
        peaksFFTKDE : list
            List of indices of the filtered peaks in the KDE frequency data.
        freq_cluster_id : numpy.ndarray
            Array of rounded frequency cluster identifiers based on the filtered peaks.
        """
        # NOTE: for now, the prominence is computed only in phase 1, and then used also in phase 2, what is changing is the bw in phase 2

        self.KDEPROMINENCE = KDEPROMINENCE  # 0. means automatic, otherwise a user_defined float value is used
        fr_diff_threshold = self.modal_cluster_resolution[
            "fr_diff_threshold"
        ]  # Determine precision if two frequencies can be considered different cluster based on a threshold
        signifiant_digits_cluster_fr_keys = self.modal_cluster_resolution[
            "signifiant_digits_cluster_fr_keys"
        ]  # Determine the number of significant digits to round the frequency cluster key

        if abs(self.KDEPROMINENCE) < np.finfo(float).eps:
            beta_plotting = True  # plot the beta distribution fitted on KDE data
        else:
            beta_plotting = False

        data = self.get_aggregated_data_format()

        data = data[
            :, np.argsort(data[0, :])
        ]  # sort data by frequency in increasing order

        # KDE along frequency
        KDE = FFTKDE(kernel="gaussian", bw="ISJ").fit(data[0, :])
        self.bw = KDE.bw
        kde_fr_x, kde_fr_y = KDE.evaluate(
            int(self.SingleSetup.fs / 2 / self.plt_resolution["freq_kde"])
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

        # check if peaks are too close to be gathered in the same cluster or not, according to fr_diff_threshold and +-bw
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

        self.current_freq_cluster_id = np.round(
            np.round(kde_fr_x[filtered_peaks] / fr_diff_threshold) * fr_diff_threshold,
            max(
                len(str(fr_diff_threshold).strip().split(".")[1]),
                signifiant_digits_cluster_fr_keys,
            ),
        )

        # PLOTTING PART

        if beta_plotting:
            fig, ax = plt.subplots(
                1,
                2,
                figsize=(10, 4),
                tight_layout=True,
                gridspec_kw={"width_ratios": [1, 3]},
            )

            beta_x_plot = np.linspace(0, 1, 100)
            beta_fitted = beta.pdf(
                beta_x_plot, alpha_par, beta_par, loc=a, scale=bminusa
            )

            ax[0].set_title("Fitted Beta Distribution")
            ax[0].plot(
                beta_fitted,
                beta_x_plot,
                color="red",
                label="Beta PDF",
            )
            ax[0].plot(
                [
                    0,
                    np.max(beta_fitted[~np.ma.masked_invalid(beta_fitted).mask]),
                ],
                [self.KDEPROMINENCE, self.KDEPROMINENCE],
                "--",
                color="black",
                label=f"Prom. Thr. {100*beta_distribution_percentile:.2f}$\%$",
            )
            ax[0].set_ylim(0, 1)
            ax[0].set_xlim(
                0,
                np.max(beta_fitted[~np.ma.masked_invalid(beta_fitted).mask]),
            )
            ax[0].invert_xaxis()
            ax[0].legend()

            ax_kde = ax[1]

        else:
            fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
            ax_kde = ax

        ax_kde.set_title(
            f"KDE along Frequency (Total No Sim. {len(self.sim_results):d})"
        )
        ax_kde.set_ylabel("Normalized KDE [-]")
        ax_kde.set_xlabel("Frequency [Hz]")
        ax_kde.plot(
            kde_fr_x, kde_fr_ynorm, color="blue", label=f"KDE (bw={self.bw:.2e})"
        )
        ax_kde.plot(
            kde_fr_x[peaksFFTKDE],
            kde_fr_ynorm[peaksFFTKDE],
            "ro",
            markersize=7,
            label=f"Sel. Peaks (prom. > {self.KDEPROMINENCE:.2e})",
        )
        ax_kde.legend()
        ax_kde.set_ylim(0, 1)
        ax_kde.set_xlim(0, self.SingleSetup.fs / 2)

        print(
            f"Computed KDE along natural frequency (No simulations so far {len(self.sim_results):d})."
        )
        logging.info(
            f"KDE natural frequency cluster (No simulations so far {len(self.sim_results):d})."
        )

        return fig, ax

    # %% CREATE AND UPDATE MODAL CLUSTERS
    def reset_clusters(self):
        self.clusters = {}
        self.clusters_id = []

    def first_time_modal_clusters_assignment(
        self,
        plot_clusters_damp_kde: bool = True,
    ):
        data = self.get_aggregated_data_format()

        fig_lists = []

        if len(list(self.clusters.keys())) == 0:
            for freq_id in self.current_freq_cluster_id:
                fig, ax, cluster1 = self._kde_damping_selection_for_a_single_cluster(
                    data, freq_id, plot_clusters_damp_kde
                )

                fig_lists.append([fig, ax])

                # construct masking boolean matrix to select poles within the list of sim_results to be associated to every cluster

                mask_list_for_each_cluster = []
                for ii, res in enumerate(self.sim_results):
                    bool_mask_for_sim = np.zeros(res["Fn_poles"].shape, dtype=bool)
                    idx_true = cluster1[
                        4:, (cluster1[3] == ii)
                    ]  # boolean indexing mask
                    bool_mask_for_sim[
                        idx_true[0, :].astype(int), idx_true[1, :].astype(int)
                    ] = True
                    mask_list_for_each_cluster.append(bool_mask_for_sim)

                self.clusters[freq_id] = mask_list_for_each_cluster
                self.clusters_id.append(freq_id)

            print(
                f"Modal Cluster assigned for the first time (No simulations so far {len(self.sim_results):d})."
            )
            logging.info(
                f"Modal Cluster assigned for the first time (No simulations so far {len(self.sim_results):d})."
            )

        return fig_lists

    def clusters_updates(
        self,
        plot_clusters_damp_kde: bool = False,
    ):
        fr_diff_threshold = self.modal_cluster_resolution[
            "fr_diff_threshold"
        ]  # Determine precision if two frequencies can be considered different cluster based on a threshold
        existing_freq_keys = np.array(list(self.clusters.keys()))
        fig_lists = []

        data = self.get_aggregated_data_format()

        for freq_id in self.current_freq_cluster_id:
            # check before if frequency are different from existing keys
            differenze_current_freq_id_existing_freq_keys = np.abs(
                freq_id - existing_freq_keys
            )

            min_value = np.min(differenze_current_freq_id_existing_freq_keys)
            min_index = np.argmin(differenze_current_freq_id_existing_freq_keys)

            if min_value < max(fr_diff_threshold, self.bw):
                # if frequency is already in the list of clusters within the threshold, then check also the MAC of the mode shapes, if MAC > 90% then assign to the same cluster the new results
                fig, ax, cluster1 = self._kde_damping_selection_for_a_single_cluster(
                    data, freq_id, plot_clusters_damp_kde
                )

                fig_lists.append([fig, ax])

                # construct masking boolean matrix to select poles within the list of sim_results to be associated to every cluster

                mask_list_for_each_cluster = []
                for ii, res in enumerate(self.sim_results):
                    bool_mask_for_sim = np.zeros(res["Fn_poles"].shape, dtype=bool)
                    idx_true = cluster1[
                        4:, (cluster1[3] == ii)
                    ]  # boolean indexing mask
                    bool_mask_for_sim[
                        idx_true[0, :].astype(int), idx_true[1, :].astype(int)
                    ] = True
                    mask_list_for_each_cluster.append(bool_mask_for_sim)

                mode_shape_for_a_mask_list = (
                    self.get_overlapped_mode_shapes_for_a_mask_list(
                        mask_list_for_each_cluster
                    )
                )

                existing_mode_shape = (
                    self.get_overlapped_mode_shape_for_an_existing_cluster(
                        list(self.clusters.keys())[min_index]
                    )
                )

                if (
                    MAC(
                        np.mean(mode_shape_for_a_mask_list, axis=0),
                        np.mean(existing_mode_shape, axis=0),
                    )
                    < self.modal_cluster_resolution[
                        "MAC_lower_threshold_to_separate_clusters"
                    ]
                ):
                    # assign freq_id to a new cluster
                    self.clusters[freq_id] = mask_list_for_each_cluster
                    self.clusters_id.append(freq_id)
                else:
                    # assign freq_id to the existing cluster
                    self.clusters[existing_freq_keys[min_index]] = (
                        mask_list_for_each_cluster
                    )
            else:
                # if frequency is not already in the list of clusters within the threshold, then create a new cluster
                fig, ax, cluster1 = self._kde_damping_selection_for_a_single_cluster(
                    data, freq_id, plot_clusters_damp_kde
                )

                fig_lists.append([fig, ax])

                # construct masking boolean matrix to select poles within the list of sim_results to be associated to every cluster

                mask_list_for_each_cluster = []
                for ii, res in enumerate(self.sim_results):
                    bool_mask_for_sim = np.zeros(res["Fn_poles"].shape, dtype=bool)
                    idx_true = cluster1[
                        4:, (cluster1[3] == ii)
                    ]  # boolean indexing mask
                    bool_mask_for_sim[
                        idx_true[0, :].astype(int), idx_true[1, :].astype(int)
                    ] = True
                    mask_list_for_each_cluster.append(bool_mask_for_sim)

                    self.clusters[freq_id] = mask_list_for_each_cluster
                    self.clusters_id.append(freq_id)

        # final check if the some keys of clusters have different dimension, fill them with False masks to have the same number of simulations, this means that one cluster have been lost during the analysis
        for freq_id in self.clusters.keys():
            if len(self.clusters[freq_id]) < len(self.sim_results):
                for ii in range(len(self.clusters[freq_id]), len(self.sim_results)):
                    bool_mask_for_sim = np.zeros(
                        self.sim_results[ii]["Fn_poles"].shape, dtype=bool
                    )
                    self.clusters[freq_id].append(bool_mask_for_sim)

        print(
            f"Modal clusters selection completed (No simulations so far {len(self.sim_results):d})."
        )
        logging.info(
            f"Modal clusters selection completed (No simulations so far {len(self.sim_results):d})."
        )

        return fig_lists

    def _kde_damping_selection_for_a_single_cluster(
        self, data, freq_id, plot_clusters_damp_kde
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
                / self.plt_resolution["damp"]
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
                >= kde_xi_x[np.argmax(kde_xi_ynorm)] - self.plt_resolution["damp"]
            )
            & (
                cluster1[2, :]
                <= kde_xi_x[np.argmax(kde_xi_ynorm)] + self.plt_resolution["damp"]
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
                [kde_xi_x[np.argmax(kde_xi_ynorm)] - self.plt_resolution["damp"]] * 2,
                [0, 1],
                "r--",
                label="Retaining Bands",
            )
            plt.plot(
                [kde_xi_x[np.argmax(kde_xi_ynorm)] + self.plt_resolution["damp"]] * 2,
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

    # %% EXTRACT MODE SHAPES FOR CLUSTERS

    def get_overlapped_mode_shapes_for_a_mask_list(self, mask_list_for_each_cluster):
        mode_shape_for_a_mask_list = []
        for ii, res in enumerate(self.sim_results):
            # pass
            mask = mask_list_for_each_cluster[ii]
            if np.sum(mask) > 0:
                mode_shape_for_a_mask_list.append(res["Phi_poles"][mask, :])

        mode_shape_for_a_mask_list = np.concatenate(mode_shape_for_a_mask_list)
        return mode_shape_for_a_mask_list

    def get_overlapped_mode_shape_for_an_existing_cluster(self, freq_id):
        mode_shape_for_selected_cluster = []
        for ii, res in enumerate(self.sim_results):
            if ii < len(self.clusters[freq_id]):
                mask = self.clusters[freq_id][ii]
                if np.sum(mask) > 0:
                    mode_shape_for_selected_cluster.append(res["Phi_poles"][mask, :])

        mode_shape_for_selected_cluster = np.concatenate(
            mode_shape_for_selected_cluster
        )
        return mode_shape_for_selected_cluster

    def get_overlapped_mode_shapes_for_all_clusters(self):
        mode_shapes_clusters = {key: [] for key in self.clusters.keys()}
        for ii, res in enumerate(self.sim_results):
            for ff, freq_id in enumerate(list(self.clusters.keys())):
                mask = self.clusters[freq_id][ii]
                if np.sum(mask) > 0:
                    mode_shapes_clusters[freq_id].append(res["Phi_poles"][mask, :])

        for ff, freq_id in enumerate(list(self.clusters.keys())):
            mode_shapes_clusters[freq_id] = np.concatenate(
                mode_shapes_clusters[freq_id]
            )
        return mode_shapes_clusters

    # %% COMPUTE IC METRIC
    def compute_IC_metric(self):
        IC_val_list = []
        # Compute IC for each simulation
        for ii, res in enumerate(self.sim_results):
            num_poles_within_retaining_bands_for_sim = 0
            for freq_id in list(self.clusters.keys()):
                num_poles_within_retaining_bands_for_sim += np.sum(
                    self.clusters[freq_id][ii]
                )
            num_stable_poles_for_sim = np.sum(
                ~np.isnan(res["Fn_poles"])
            )  # np.sum(~np.isnan(res['Fn_poles']))
            IC_computed = (
                num_poles_within_retaining_bands_for_sim / num_stable_poles_for_sim
            )
            self.sim_results[ii]["IC"][1] = IC_computed
            IC_val_list.append(IC_computed)
        return IC_val_list

    def plot_ic_graph(self, sim_step: int = 5):
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
        ax.set_xticks(range(0, len(self.sim_results), sim_step))

        return fig, ax

    # %% DUMP RESULTS TO FILES
    def dump_last_results_to_file(self, output_path):
        with open(
            output_path + os.sep + f"{len(self.Results.sim_results):d}.pkl", "wb"
        ) as backup_file:
            pickle.dump(self.Results.sim_results, backup_file)

    #  SIMPLIFIED!!! NOT NECESSARY IN THE FINAL CODE
    def visualize_mode_shape_from_clusters(self, stddev_fct: float = 3.0):
        mode_shapes_clusters = self.get_overlapped_mode_shapes_for_all_clusters()

        fig_lists = []

        for ff, freq_id in enumerate(list(self.clusters.keys())):
            fig_lists.append(plt.figure())
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
                modo_medio - stddev_fct * modo_std_dev,
                modo_medio + stddev_fct * modo_std_dev,
                alpha=0.2,
                color="C1",
            )
            plt.legend()
            plt.title(f"Mode Shape Cluster at {freq_id:.2f} Hz")
            plt.xlabel("Mode Shape Index")
            plt.ylabel("Amplitude")
            plt.tight_layout()

        return fig_lists

        # NOTE: test checking masking: check the number of trues with np.sum(self.clusters[freq_id][ii]) , mask = self.clusters[freq_id][ii]
        # list_sim_results[0]['Fn_poles'][mask]
        # list_sim_results[0]['Xi_poles'][mask]
        # list_sim_results[0]['Phi_poles'][mask]
