import numpy as np
import logging
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import os
# import mplcursors
# from mpldatacursor import datacursor

from .helper import run_SSICov_with_timeout

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
                            self.progressive_plot_overlap_stab_diag(
                                self.sim_results,
                                plt_resolution=plt_resolution,
                                sim=sim,
                                update_flag=False,
                            )
                        )
                        plt.pause(0.001)
                        plt.savefig(
                            f"{self.output_path_phase1}/Overlap_Stab_Diag_until_sim_{1000+sim+1:d}.png",
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
                            self.progressive_plot_overlap_stab_diag(
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
                            f"{self.output_path_phase1}/Overlap_Stab_Diag_until_sim_{1000+sim+1:d}.png",
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
                        self.progressive_plot_overlap_stab_diag(
                            self.sim_results,
                            plt_resolution=plt_resolution,
                            sim=Nsim_batch - 1,
                            update_flag=False,
                        )
                    )
                    plt.pause(0.001)
                    plt.savefig(
                        f"{self.output_path_phase1}/Overlap_Stab_Diag_until_sim_{1000+Nsim_batch-1:d}.png",
                        dpi=200,
                    )

                    # TODO: implement a function to update the heatmap related to damping vs freq cluster
                else:
                    # update_overlap_stab_diag()
                    new_sim_results = []
                    for ii in range(sim, sim + Nsim_batch):
                        new_sim_results.append(self.sim_results[ii])
                    fig, ax, heatmap, xedges, yedges, im = (
                        self.progressive_plot_overlap_stab_diag(
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
                        f"{self.output_path_phase1}/Overlap_Stab_Diag_until_sim_{1000+sim+Nsim_batch-1:d}.png",
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

    def progressive_plot_overlap_stab_diag(
        self,
        list_sim_results,
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
            def update_heatmap(new_x, new_y, heatmap, im, xedges, yedges):
                # global heatmap
                new_heatmap, _, _ = np.histogram2d(
                    new_x, new_y, bins=[xedges, yedges]
                )  # np.histogram2d(new_x, new_y)#, \
                # bins=[round(SingleSetup.fs/2/plt_resolution['freq']),int(new_x[1,:].max()/plt_resolution['order'])], \
                # range = [[0, SingleSetup.fs/2], [new_x[1,:].min(), new_x[1,:].max()]])
                heatmap += new_heatmap
                heatmap = heatmap / np.max(
                    heatmap
                )  # normalize the heatmap between 0 and 1
                im.set_data(heatmap.T)
                plt.draw()
                return im, heatmap

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
        plt_resolution: dict = {"freq": 1, "damp": 0.001, "order": 1},
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
        # annot = ax.annotate("", xy=(0,0), xytext=(10,10),
        #                     textcoords="offset points",
        #                     bbox=dict(boxstyle="round", fc="w"),
        #                     arrowprops=dict(arrowstyle="->"))
        # annot.set_visible(False)

        # def update_annot(event):
        #     if event.inaxes == ax:
        #         x, y = event.xdata, event.ydata
        #         annot.xy = (x, y)
        #         text = f"(f={x:.2f} Hz, xi={y*100:.2f} %)"
        #         annot.set_text(text)
        #         annot.set_visible(True)
        #         fig.canvas.draw_idle()

        # def on_click(event):
        #     if event.inaxes == ax:
        #         update_annot(event)
        #     else:
        #         annot.set_visible(False)
        #         fig.canvas.draw_idle()

        # fig.canvas.mpl_connect("button_press_event", on_click)

        return fig, ax

    # def plot_overlap_damp_order_cluster(self, method: str ='density', plt_resolution : dict ={'freq': 1, 'damp': 0.001, 'order': 1}):
    #     """
    #     Plot Overlapped Damping Cluster Diagram
    #     x = Order
    #     y = Xi_poles

    #     Input:
    #     method: str = 'density' or 'scatter', default='density'
    #     """
    #     step = 1
    #     x = []
    #     y= []
    #     for ii, res in enumerate(self.sim_results):
    #         Xi_stab = np.where(res['Lab'] == 1, res['Xi_poles'], np.nan)
    #         x.extend(list(np.array([i // len(Xi_stab) for i in range(Xi_stab.flatten(order="f").shape[0])]) * step))
    #         y.extend(list(Xi_stab.flatten(order="f")))

    #     data=np.array([x, y])
    #     data = data[:,~np.isnan(data).any(axis=0)]

    #     if method == 'scatter':
    #         fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    #         ax.set_title(f"Overlap. Damping Order Cluster (i-AOMA Phase 1, Nsim={self.NsimPh1:d})")
    #         ax.set_ylabel("Damping Ratio [-]")
    #         ax.set_xlabel("Model Order [-]")
    #         ax.plot(data[0,:], data[1,:], "go", markersize=7, alpha=0.1)
    #         # mplcursors.cursor()
    #     elif method == 'density':
    #         fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    #         ax.set_title(f"Overlap. Damping Order Cluster Density Heatmap (i-AOMA Phase 1, Nsim={self.NsimPh1:d})")
    #         ax.set_ylabel("Damping Ratio [-]")
    #         ax.set_xlabel("Model Order [-]")
    #         Z, xedges, yedges = np.histogram2d(data[0,:], data[1,:], bins=[round(data[0,:].max()/plt_resolution['order']),int(round(data[1,:].max(),3)/plt_resolution['damp'])])
    #         heatmap_plot = plt.pcolormesh(xedges, yedges, Z.T)
    #         fig.colorbar(heatmap_plot, ax=ax)
    #     annot = ax.annotate("", xy=(0,0), xytext=(10,10),
    #                         textcoords="offset points",
    #                         bbox=dict(boxstyle="round", fc="w"),
    #                         arrowprops=dict(arrowstyle="->"))
    #     annot.set_visible(False)

    #     def update_annot(event):
    #         if event.inaxes == ax:
    #             x, y = event.xdata, event.ydata
    #             annot.xy = (x, y)
    #             text = f"(Order={x:.0f} Hz, xi={y*100:.2f} %)"
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

    #     return fig, ax

    # def overlap_mode_shapes(self, plt_resolution : dict ={'freq': 1, 'damp': 0.001, 'order': 1}):
    #     """
    #     Plot the overlap mode shape of the poles
    #     """
    #     step = 1
    #     freq, order, damping, sim_id = [], [], [], []
    #     for ii, res in enumerate(self.sim_results):
    #         # print(res['Fn_poles'].shape)
    #         Fns_stab = np.where(res['Lab'] == 1, res['Fn_poles'], np.nan)
    #         Xi_stab = np.where(res['Lab'] == 1, res['Xi_poles'], np.nan)
    #         freq.extend(list(Fns_stab.flatten(order="f")))
    #         order.extend(list(np.array([i // len(Fns_stab) for i in range(Fns_stab.flatten(order="f").shape[0])]) * step))
    #         damping.extend(list(Xi_stab.flatten(order="f")))
    #         sim_id.extend([res['NumAnal']]*len(Fns_stab.flatten(order="f")))
    #     data=np.array([freq, order, damping, sim_id])
    #     data = data[:,~np.isnan(data).any(axis=0)]

    #     fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    #     ax.set_title(f"Overlap. Stab. Diag. (i-AOMA Phase 1, Nsim={self.NsimPh1:d})")
    #     ax.set_ylabel("Model Order [-]")
    #     ax.set_xlabel("Frequency [Hz]")
    #     scatter_2D_stab_diag = ax.scatter(data[0,:], data[1,:], c=data[2, :], cmap='viridis' , marker='o',alpha=0.1)
    #     # Add a color bar
    #     cbar = fig.colorbar(scatter_2D_stab_diag, ax=ax, pad=0.1)
    #     cbar.set_label('Damping')

    #     fig = plt.figure(figsize=(10, 7))
    #     ax = fig.add_subplot(111, projection='3d')
    #     scatter_3d_plot = ax.scatter(data[0, :], data[1, :], data[2, :], c=data[3, :], cmap='viridis', marker='o', alpha=0.1)

    #     ax.set_xlabel('Frequency [Hz]')
    #     ax.set_ylabel('Order')
    #     ax.set_zlabel('Damping Ratio [-]')
    #     ax.set_title('Overlap Mode Shapes')

    #     # Add a color bar
    #     cbar = fig.colorbar(scatter_3d_plot, ax=ax, pad=0.1)
    #     cbar.set_label('Simulation ID')

    #     # Reverse the direction of the y-axis
    #     ax.invert_yaxis()
    #     # ax.view_init(30, -60, 0) # default 3d view
    #     ax.view_init(elev=90, azim=90, roll=0) # view simialr to stab diagram
    #     ax.view_init(elev=0, azim=90, roll=0) # view simialr to freq-damping cluster diagram
    #     plt.draw()

    #     Z_freq_order, xedges_freq_order, yedges_freq_order = np.histogram2d(data[0,:], data[1,:], \
    #                                            bins=[round(self.SingleSetup.fs/2/plt_resolution['freq']),int(data[1,:].max()/plt_resolution['order'])], \
    #                                            range = [[0, self.SingleSetup.fs/2], [data[1,:].min(), data[1,:].max()]])
    #     Z_freq_damp, xedges_freq_damp, yedges_freq_damp = np.histogram2d(data[0,:], data[2,:], \
    #                                            bins=[round(self.SingleSetup.fs/2/plt_resolution['freq']),int(round(data[2,:].max(),3)/plt_resolution['damp'])], \
    #                                            range = [[0, self.SingleSetup.fs/2], [data[2,:].min(), data[2,:].max()]])
    #     # fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    #     # heatmap_plot = plt.pcolormesh(xedges_freq_damp, yedges_freq_damp, Z_freq_damp.T)

    #     # To numerically estimate the joint probability over the 3D space (freq, order, damp) from the two bivariate marginals (Z_freq_order and Z_freq_damp), you can use the concept of copulas or a simpler approach by assuming independence between the marginals. Here, I'll demonstrate a simpler approach assuming independence.

    #     # Normalize the marginals to ensure they sum to 1
    #     Z_freq_order /= Z_freq_order.sum()
    #     Z_freq_damp /= Z_freq_damp.sum()

    #     # Estimate the joint probability assuming independence
    #     Z_joint = np.zeros((Z_freq_order.shape[0], Z_freq_order.shape[1], Z_freq_damp.shape[1]))

    #     for i in range(Z_freq_order.shape[0]):
    #         for j in range(Z_freq_order.shape[1]):
    #             for k in range(Z_freq_damp.shape[1]):
    #                 Z_joint[i, j, k] = Z_freq_order[i, j] * Z_freq_damp[i, k]

    #     # Normalize the joint probability to ensure it sums to 1
    #     Z_joint /= Z_joint.sum()

    #     # Create meshgrids for the 3D space
    #     F, O, D = np.meshgrid([tmp for tmp in np.arange(0, self.SingleSetup.fs/2, plt_resolution['freq'])],
    #                           [tmp for tmp in np.arange(0, data[1,:].max(), plt_resolution['order'])],
    #                           [tmp for tmp in np.arange(0, data[2,:].max(), plt_resolution['damp'])],
    #                           indexing='ij')

    #     # Flatten the meshgrid and Z_joint arrays
    #     F_flat = F.flatten()
    #     O_flat = O.flatten()
    #     D_flat = D.flatten()
    #     Z_joint_flat = Z_joint.flatten()

    #     density_over_zero = np.histogram(Z_joint_flat)[1][1]

    #     # Plot the scatter plot
    #     fig = plt.figure(figsize=(10, 7))
    #     ax = fig.add_subplot(111, projection='3d')
    #     scatter = ax.scatter(F_flat[Z_joint_flat>density_over_zero], O_flat[Z_joint_flat>density_over_zero], D_flat[Z_joint_flat>density_over_zero], c=Z_joint_flat[Z_joint_flat>density_over_zero], cmap='Blues', marker='o',alpha=1.0)

    #     ax.set_xlabel('Frequency [Hz]')
    #     ax.set_ylabel('Order')
    #     ax.set_zlabel('Damping Ratio [-]')
    #     ax.set_title('Joint Probability Distribution')

    #     # Add a color bar
    #     cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    #     cbar.set_label('Joint Probability')
