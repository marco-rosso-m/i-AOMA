"""
Esempio su dati trave legno
"""

# %% Import modules and set up input data
import numpy as np
import pandas as pd

# from pyoma2.algorithms import FSDD, SSIcov, pLSCF
# from pyoma2.setup import SingleSetup
import os
import matplotlib.pyplot as plt

# matplotlib.get_backend()
# matplotlib.use('tkagg')
# import mplcursors
# from pyoma2.functions.plot import plot_mac_matrix


from i_aoma.ver_2_0.IAOMASingleSetup import IAOMASingleSetup

plt.plot([0, 0])
plt.show()
plt.close()

data_path = "src/i_aoma/ver_2_0/test_data"
data_filename = "TRAVE1(AF, 1cuscino)_Job1_2020_05_28_07_16_09_001_001"
data_filename_ext = ".xlsx"
# FreQ = [7.10, 8.69, 14.09]
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
Timber_ss = IAOMASingleSetup(data, fs=fs)

_geo1 = data_path + os.sep + "Geo1_timber.xlsx"
_geo2 = data_path + os.sep + "Geo2_timber.xlsx"

Timber_ss.def_geo(_geo1, _geo2)

fig, ax = Timber_ss.ss.plot_geo1()

print("ok")
