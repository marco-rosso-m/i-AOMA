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
NsimPh1 = 9
# Number of batch simulations to be done in parallel
Nsim_batch = 3
# Number of core to use for parallel computation (-1 to use all available, 0 to disable parallel computation)
n_jobs = 1
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

# %% IAOMA RUN PHASE 1 (NO GUI REQUIRED)
fig, ax = Timber_ss.run_phase1(
    NsimPh1=NsimPh1,
    n_jobs=n_jobs,
    timeout_seconds=timeout_seconds,
    Nsim_batch=Nsim_batch,
    plt_stab_diag_backup=True,
)

# _, _ = Timber_ss.plot_phase1_overlapped_stab_diag()
_, _ = Timber_ss.plot_phase1_overlapped_cluster_diag()


mplcursors.cursor()

print("End of the script")
