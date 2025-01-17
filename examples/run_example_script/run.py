"""
Example: Timber Beam i-AOMA ver 2.0

References: Pasca, D. P., Aloisio, A., Fragiacomo, M., & Tomasi, R. (2021).
            Dynamic characterization of timber floor subassemblies: Sensitivity analysis and modeling issues.
            Journal of Structural Engineering, 147(12), 05021008.

@author: Marco Martino Rosso (marco.rosso@polito.it)
"""

# %% IMPORTS
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
import mplcursors

from i_aoma.IAOMA import IAOMA

# %% USER DEFINITIONS
# data path
data_path = "examples/run_example_script/test_data"
data_filename = "TRAVE1(AF, 1cuscino)_Job1_2020_05_28_07_16_09_001_001"
data_filename_ext = ".xlsx"
# output path
output_path = "examples/run_example_script/Results/trave1_results_1cuscino_17_01_25"
# Sampling Frequency [Hz]
fs = 1200
# fundamental frequency [Hz]
fundfreq = 65.98
# Number of simulations to be done in phase 1
NsimPh1 = 20
# Number of batch simulations to be done in parallel
Nsim_batch = 10
# Number of core to use for parallel computation (-1 to use all available, 0 to disable parallel computation)
n_jobs = -1
# time out for each simulation in phase 1 [s]
timeout_seconds = 30

# %% PRE-INITIALIZATION
# set matplotlib backend
matplotlib.get_backend()
matplotlib.use("qtagg")
plt.plot([0, 0])
plt.close()
# Create output folder if it does not exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# %% DATA LOADING and conversion to NumPy array (rows: time, columns: sensors)
if not os.path.exists(data_path + os.sep + data_filename + ".parquet"):
    print("Data file not found, creating it...")
    data = pd.read_excel(
        data_path + os.sep + data_filename + data_filename_ext, header=None
    ).dropna()
    data.to_parquet(data_path + os.sep + data_filename + ".parquet")

data = pd.read_parquet(data_path + os.sep + data_filename + ".parquet").to_numpy()

# %% IAOMA SETUP
# create instance of iAOMA setup
Timber_ss = IAOMA(
    data,
    fs=fs,
    ff=fundfreq,
    output_path=output_path,
)
# freely modify the default limits
Timber_ss.manual_update_sampling_limits(ordmax=200)
# print the currently set sampling limits
Timber_ss.print_qmc_sampling_limits()

Timber_ss.preprocess_data(
    DecFct=0,
    detrend=True,
)

# Timber_ss.load_phase1_from_file(
#     [
#         output_path + os.sep + "Phase1" + os.sep + "phase1_metadata.pkl",
#         output_path + os.sep + "Phase1" + os.sep + "phase1_results.pkl",
#     ]
# )

# # %% IAOMA RUN PHASE 1
fig, ax = Timber_ss.run_phase1(
    NsimPh_max=NsimPh1,
    n_jobs=n_jobs,
    timeout_seconds=timeout_seconds,
    Nsim_batch=Nsim_batch,
    plt_stab_diag_backup=True,
    progressive_plot_flag=True,
)

Timber_ss.Results.plot_overlap_stab_diag()
Timber_ss.Results.plot_overlap_freq_damp_cluster()
plt.ylim(0, 0.02)
plt.xlim(0, 600)
mplcursors.cursor()


_, _ = Timber_ss.Results.normalized_kde_frequency_filtering()
mplcursors.cursor()

_ = Timber_ss.Results.first_time_modal_clusters_assignment()
# to clear clusters it is possible to use the following command: Timber_ss.Results.clusters = {}
print(Timber_ss.Results.clusters_id)


IC_val_list = Timber_ss.Results.compute_IC_metric()
_, _ = Timber_ss.Results.plot_ic_graph(sim_step=1)

# %% SIMULATE NEW RESULTS
# Timber_ss.Results.add_new_sim_results(Timber_ss.Results.sim_results)
# print(len(Timber_ss.Results.sim_results))
# Timber_ss.Results.normalized_kde_frequency_filtering(KDEPROMINENCE=Timber_ss.Results.KDEPROMINENCE)

# _ = Timber_ss.Results.first_time_modal_clusters_assignment()
# natfreqs = list(Timber_ss.Results.clusters.keys())
# print(len(Timber_ss.Results.clusters[natfreqs[0]]),'simulations')

# _ = Timber_ss.Results.clusters_updates(plot_clusters_damp_kde=True)

# Timber_ss.Results.visualize_mode_shape_from_clusters()


# TODO: funzione da rivedere....
# Timber_ss.phase1.visualize_qmc_samples_distribution()


# for ii in plt.get_fignums():
#     print(ii)
#     plt.figure(ii)
#     plt.savefig(output_path + os.sep + f"plot_{ii}.png", dpi=300, bbox_inches="tight")

# Timber_ss.dump_phase1_to_file()


# plt.close("all")

# TODO: implement controls on every function to ensure that user follow the correct order of operations

# TODO: implement a function to save the results of phase 1 in a file (two functions, one for dumping
# the results after run_phase1 and one for loading the results before processing them at the end of phase1)
# and other two functions to save and load processed results, interms of clusters and IC values, ready to start phase 2
# with the RF training


mode_shapes_clusters = Timber_ss.Results.get_overlapped_mode_shapes_for_all_clusters()
freq_id = Timber_ss.Results.clusters_id
mode_shapes_clusters_mean = [
    np.mean(mode_shapes_clusters[freq_id[ii]], axis=0) for ii in range(len(freq_id))
]
mode_shapes_clusters_std = [
    np.std(mode_shapes_clusters[freq_id[ii]], axis=0) for ii in range(len(freq_id))
]
samplecovmat = [
    np.trace(np.cov(mode_shapes_clusters[freq_id[ii]].T).real)
    for ii in range(len(freq_id))
]
print(samplecovmat)

rf_model = Timber_ss.rf_intelligent_core_training()


NsimPh2 = 200
Nsim_batch2 = 50  # batch of analysis and for convergence check
n_jobs2 = 1

Timber_ss.run_phase2(
    NsimPh_max=NsimPh2,
    n_jobs=n_jobs2,
    timeout_seconds=timeout_seconds,
    Nsim_batch=Nsim_batch2,
    plt_stab_diag_backup=True,
    progressive_plot_flag=False,
)

# Timber_ss.run_phase2(
#     NsimPh2=NsimPh2,
#     n_jobs=n_jobs,
#     timeout_seconds=timeout_seconds,
#     Nsim_batch=Nsim_batch2,
#     Nsim_batch_conv_check = Nsim_batch_conv_check,
#     plt_stab_diag_backup=True,
# )


print("End of phase 2")

IC_val_list = Timber_ss.Results.compute_IC_metric()
_, _ = Timber_ss.Results.plot_ic_graph()


print("End of the script")
