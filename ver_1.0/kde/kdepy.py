from kde.helpers import *

from KDEpy import FFTKDE
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import beta
import numpy as np


class Kde():
    def __init__(self, jointpolesmodes, fs, KDEPROMINENCE):
        # columns ['Frequency', 'Order', 'Label', 'Damp', 'Emme', 'ModeNum', 'SimNumber',   'SimNumber','dof','dof','...' ]
        self.jointpolesmodes = jointpolesmodes
        self.fs = fs
        self.KDEPROMINENCE = KDEPROMINENCE
        # Sort the entire array according to the first column
        self.jointpolesmodes = self.jointpolesmodes[self.jointpolesmodes[:, 0].argsort()]
        # Train KDE with automatic bw ISJ algorithm
        self.KDE = FFTKDE(kernel='gaussian', bw='ISJ').fit(self.jointpolesmodes[:,0])
        self.bw = self.KDE.bw
        self.x, self.y = self.KDE.evaluate(int(self.fs/2*1000))
        self.ynorm = self.y / max(self.y)

        if self.KDEPROMINENCE == 'automatic':
            # automatic prominence definition based on confidence interval of a beta distribution
            # fitted on the KDE data
            beta_fit = beta.fit(self.ynorm, floc=0, fscale=1.001)
            self.alpha_par, self.beta_par, self.a, self.bminusa = beta_fit
            self.KDEPROMINENCE = beta.ppf(0.99, self.alpha_par, self.beta_par, loc=self.a, scale=self.bminusa)
        # find peaks according to prominence threshold
        self.peaksFFTKDE, _ = find_peaks(self.ynorm, prominence=self.KDEPROMINENCE) 
        

    def select_modes_clusters(self):
        self.KDEbwFactor = 0
        is_full = 1
        while is_full : # any() function returns True if any list within the list is not empty
            is_full=0
            self.KDEbwFactor += 1 # increment bandwidth multiplier until all the frequency clusters selected at least one poles
            self.Freqinter=[] # necessary to reinitialize if a new increment self.KDEbwFactor is considered
            # define frequency bands around peaks of KDE
            for ii in range(len(self.peaksFFTKDE)):
                idx_pre =  (np.abs(self.x - [self.x[self.peaksFFTKDE[ii]] - self.KDEbwFactor * self.bw])).argmin()
                idx_post = (np.abs(self.x - [self.x[self.peaksFFTKDE[ii]] + self.KDEbwFactor * self.bw])).argmin()
                self.Freqinter.append(np.array([self.x[idx_pre],self.x[idx_post]]))

            self.Frequency_dataset=[] # necessary to reinitialize if a new increment self.KDEbwFactor is considered

            for ii in range(len(self.Freqinter)):# selecting rows of poles around the KDE peaks
                sel_rows_df = np.array([], dtype=np.float64)
                sel_rows_df = self.jointpolesmodes[self.jointpolesmodes[:,0] > self.Freqinter[ii][0], :]
                sel_rows_df = sel_rows_df[sel_rows_df[:,0] < self.Freqinter[ii][1], :]
                self.Frequency_dataset.append(sel_rows_df)
                if not sel_rows_df.size:
                    is_full=1
                    print(f'Cluster {ii:d} is empty with {self.KDEbwFactor:d} times of the bandwidth of {self.bw:.4f} Hz, try to increase the bandwith...\n')
        

    def plot_select_modes_clusters(self, RESULTS_PATH):
        # Plotting selected frequency clusters
        plot_selected_frequency_clusters(self,RESULTS_PATH)

    
    def information_content(self, selectedpoles_totnum):
        self.effective_selectedpoles_totnum = count_num_effective_poles_for_each_simulation(selectedpoles_totnum, self.Frequency_dataset)
        self.IC = np.divide(np.array(self.effective_selectedpoles_totnum, dtype=np.int32), np.array(selectedpoles_totnum, dtype=np.int32))
        return self.IC

    def save_plot_IC(self, MAX_NUM_MC_SIM, ICTHRESH, RESULTS_PATH):
        plot_IC_graph(self.IC, MAX_NUM_MC_SIM, ICTHRESH, RESULTS_PATH)

    def plot_kde_freq(self, RESULTS_PATH):
        fig1, ax1 = plt.subplots(figsize=(10,3))
        plt.plot(self.x, self.ynorm, label='KDE',lw=2,color='#023e7d')
        plt.plot(self.x[self.peaksFFTKDE], self.ynorm[self.peaksFFTKDE], color='r', linestyle='None', marker='2',ms=15,lw=10)
        # Add markers and corresponding text by looping through the 
        for pp, qq in zip(self.x[self.peaksFFTKDE],self.ynorm[self.peaksFFTKDE]):
            ax1.text(pp-0.2, qq+0.1, f"{pp:.2f}Hz")
        plt.title('Stable poles KDE along Frequency',fontweight='bold')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Normalized KDE [-]')
        legend_elements = [ Line2D([0], [0], marker=None, color='#023e7d', lw=2, label=f'Normalized KDE, bw ISJ ={self.bw:.5f}'),
                            Line2D([0], [0], marker='2', ms=15, color='r', lw=0, label=f'Peaks with prominence >{self.KDEPROMINENCE:.3f}') ]
        ax1.legend(handles=legend_elements, loc='upper right',framealpha=0.95)
        plt.ylim(plt.gca().get_ylim()[0],plt.gca().get_ylim()[1]+0.20)
        plt.tight_layout()
        plt.savefig(RESULTS_PATH+f"/KDE.png", dpi=300)
        plt.close()

    def export_results_to_file(self, RESULTS_PATH):
        with open(RESULTS_PATH+'/Frequency_dataset.npy', 'wb') as f:
                np.save(f, np.array(self.Frequency_dataset, dtype='object'))
        # Test loading file
        # X = np.load(RESULTS_PATH+'/Phase1'+'/Frequency_dataset.npy', allow_pickle = True)




