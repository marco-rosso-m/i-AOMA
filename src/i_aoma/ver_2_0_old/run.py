"""
Esempio su dati trave legno
"""

# %% Import modules and set up input data
import numpy as np
import pandas as pd
import multiprocessing

import time

# from pyoma2.algorithms import FSDD, SSIcov, pLSCF
# from pyoma2.setup import SingleSetup
import os
import matplotlib.pyplot as plt

import matplotlib

from i_aoma.ver_2_0_old.IAOMA import IAOMA


matplotlib.get_backend()
matplotlib.use("tkagg")
# import mplcursors


plt.plot([0, 0])
# plt.show()
plt.close()

data_path = "src/i_aoma/ver_2_0/test_data"
data_filename = "TRAVE1(AF, 1cuscino)_Job1_2020_05_28_07_16_09_001_001"
data_filename_ext = ".xlsx"
FreQ = [65.98]
# Orders = [5,9,6]
output_path = "src/i_aoma/ver_2_0/Results/trave1_results_1cuscino"

fs = 1200  # [Hz] Sampling Frequency

FreQ = [65.6, 155.9, 273.0]
Orders = [10, 10, 10]

# %% Initialization
if not os.path.exists(output_path):
    os.makedirs(output_path)

# data = pd.read_excel(data_path, header=0, index_col=False)
# data = data.to_numpy()
if not os.path.exists(data_path + os.sep + data_filename + ".parquet"):
    print("Data file not found, creating it...")
    data = pd.read_excel(
        data_path + os.sep + data_filename + data_filename_ext, header=None
    ).dropna()
    data.to_parquet(data_path + os.sep + data_filename + ".parquet")

data = pd.read_parquet(data_path + os.sep + data_filename + ".parquet").to_numpy()

datamean = np.median(data, axis=1)
for ii in range(data.shape[1]):
    data[:, ii] = data[:, ii] - datamean[ii]


# create instance of iAOMA setup
Timber_ss = IAOMA(
    data,
    fs=fs,
    ff=FreQ[0],
    DecFct=0,
    detrend=True,
)
# freely modify the default limits
Timber_ss.ordmax = 200

Timber_ss.print_qmc_sampling_limits()


try:
    t0 = time.time()
    # Your existing code
    Timber_ss.run_phase1(NsimPh1=50, n_jobs=-1, timeout_seconds=30)
    t1 = time.time()
    print(f"Elapsed time: {t1-t0} s")

    # _geo1 = data_path + os.sep + "Geo1_timber.xlsx"
    # _geo2 = data_path + os.sep + "Geo2_timber.xlsx"
    # Timber_ss.def_geo(_geo1, _geo2)
    # fig, ax = Timber_ss.ss.plot_geo1()

    print("ok")
finally:
    # Ensure all multiprocessing resources are cleaned up
    multiprocessing.active_children()
# Timber_ss.run_phase1(NsimPh1=50, n_jobs=2, timeout_seconds=30)

# _geo1 = data_path + os.sep + "Geo1_timber.xlsx"
# _geo2 = data_path + os.sep + "Geo2_timber.xlsx"

# Timber_ss.def_geo(_geo1, _geo2)

# fig, ax = Timber_ss.ss.plot_geo1()

print("ok")
