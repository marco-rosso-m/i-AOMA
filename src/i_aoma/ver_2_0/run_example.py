"""
Example: Timber Beam i-AOMA ver 2.0

References: Pasca, D. P., Aloisio, A., Fragiacomo, M., & Tomasi, R. (2021).
            Dynamic characterization of timber floor subassemblies: Sensitivity analysis and modeling issues.
            Journal of Structural Engineering, 147(12), 05021008.

@author: Marco Martino Rosso (marco.rosso@polito.it)
"""

# %% IMPORTS
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
import mplcursors

from i_aoma.ver_2_0.IAOMA import IAOMA

# %% USER DEFINITIONS
# data path
data_path = "src/i_aoma/ver_2_0/test_data"
data_filename = "TRAVE1(AF, 1cuscino)_Job1_2020_05_28_07_16_09_001_001"
data_filename_ext = ".xlsx"
# output path
output_path = "src/i_aoma/ver_2_0/Results/trave1_results_1cuscino"
# Sampling Frequency [Hz]
fs = 1200
# fundamental frequency [Hz]
fundfreq = 65.98
# Number of simulations to be done in phase 1
NsimPh1 = 50
# Number of batch simulations to be done in parallel
Nsim_batch = 50
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
    DecFct=0,
    detrend=True,
    output_path=output_path,
)
# freely modify the default limits
Timber_ss.ordmax = 200
# print the currently set sampling limits
Timber_ss.print_qmc_sampling_limits()

Timber_ss.load_phase1_from_file(
    [
        output_path + os.sep + "Phase1" + os.sep + "phase1_metadata.pkl",
        output_path + os.sep + "Phase1" + os.sep + "phase1_results.pkl",
    ]
)

# %% IAOMA RUN PHASE 1 (NO GUI REQUIRED)
fig, ax = Timber_ss.run_phase1(
    NsimPh1=NsimPh1,
    n_jobs=n_jobs,
    timeout_seconds=timeout_seconds,
    Nsim_batch=Nsim_batch,
    plt_stab_diag_backup=True,
)

_, _ = Timber_ss.phase1.plot_overlap_stab_diag()
_, _ = Timber_ss.phase1.plot_overlap_freq_damp_cluster()

plt.ylim(0, 0.02)
plt.xlim(0, 600)

mplcursors.cursor()


_, _ = Timber_ss.phase1.normalized_kde_frequency_filtering()
mplcursors.cursor()

print(Timber_ss.phase1.freq_cluster_id)

_, _ = Timber_ss.phase1.normalized_kde_frequency_filtering(KDEPROMINENCE=0.2)
mplcursors.cursor()

print(Timber_ss.phase1.freq_cluster_id)

clusters_id = Timber_ss.phase1.kde_clusters_selection(plot_clusters_damp_kde=True)

IC_val_list = Timber_ss.phase1.compute_ic_phase1()

_, _ = Timber_ss.phase1.plot_ic_graph()

Timber_ss.phase1.visualize_qmc_samples_distribution()

Timber_ss.phase1.visualize_mode_shape_from_clusters()

for ii in plt.get_fignums():
    print(ii)
    plt.figure(ii)
    plt.savefig(output_path + os.sep + f"plot_{ii}.png", dpi=300, bbox_inches="tight")

Timber_ss.dump_phase1_to_file()


plt.close("all")

# TODO: implement controls on every function to ensure that user follow the correct order of operations

# TODO: implement a function to save the results of phase 1 in a file (two functions, one for dumping
# the results after run_phase1 and one for loading the results before processing them at the end of phase1)
# and other two functions to save and load processed results, interms of clusters and IC values, ready to start phase 2
# with the RF training


Timber_ss.phase2_start()


print("End of phase 1")


print("End of the script")
