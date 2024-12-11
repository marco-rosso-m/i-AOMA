# -*- coding: utf-8 -*-
"""

Author:
Marco Martino Rosso
"""

import numpy as np
import matplotlib.pyplot as plt


from pyoma2.setup import SingleSetup
# from pyoma2.functions.plot import plot_mac_matrix

plt.plot([0, 0])
plt.close()


class IAOMASingleSetup:
    def __init__(self, data: np.ndarray, fs: float):
        self.ss = SingleSetup(data, fs=fs)

    def def_geo(self, *arg):
        """
        arg contains path to excel file for defining geometry according to PyOMA2 standard
        _geo1,_geo2
        """
        try:
            if len(arg) == 1:
                self.ss.def_geo1_by_file(arg[0])
            else:
                self.ss.def_geo1_by_file(arg[0])
                self.ss.def_geo2_by_file(arg[1])
        except Exception:
            raise ValueError("Invalid geometry type!")
