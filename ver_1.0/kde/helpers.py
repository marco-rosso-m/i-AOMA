import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def plot_selected_frequency_clusters(self, RESULTS_PATH):
    fig1, ax1 = plt.subplots(figsize=(10,3))
    for ii in range(len(self.Freqinter)):   # col 0 = 'Frequency'                 # col 1 = 'Order'
        ax1 = sns.scatterplot(x=self.Frequency_dataset[ii][:,0], y=self.Frequency_dataset[ii][:,1]*2, 
            label=f'Mode {ii+1}: {np.mean(self.Frequency_dataset[ii][:,0]):.2f} Hz; Num. Poles: {self.Frequency_dataset[ii][:,0].shape[0]}')

    ax1.set_xlim(left=0, right=self.fs/2)
    ax1.set_ylim(bottom=0, top=max(self.Frequency_dataset[ii][:,1]*2)+10) # col 1 = 'Order'
    ax1.set_title('Extracted modes from KDE peaks $\pm$'+ f'{self.KDEbwFactor}'+ ' ' +'$bw_{ISJ}$')
    ax1.set_xlabel('Frequency [Hz]')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(RESULTS_PATH + f"/KDE_frequency_clusters.png", dpi=300)
    plt.close()

def count_num_effective_poles_for_each_simulation(selectedpoles_totnum,Frequency_dataset):
    count_num_effective_poles_sim=[]
    for kk in range(len(selectedpoles_totnum)): # kk accounts for the k-th simulation
        tmp_extracted=[]
        for jj in range(len(Frequency_dataset)):
            # for each cluster, count the num of selected stable poles 
            tmp_extracted.append( Frequency_dataset[jj][ Frequency_dataset[jj][:,6] == kk, :].shape[0] )
        # for simulation k-th count the num of retained stable poles within frequency cluster bands
        count_num_effective_poles_sim.append(sum(tmp_extracted))

    return count_num_effective_poles_sim

def plot_IC_graph(IC, MAX_NUM_MC_SIM, ICTHRESH, RESULTS_PATH):
    fig,ax=plt.subplots(figsize=(10,4))
    plt.plot(np.arange(1,MAX_NUM_MC_SIM+2), IC,
             color='#023e7d', linestyle='None', marker='.',ms=5,label='IC for Analysis actually conducted')
    plt.plot(np.arange(1,MAX_NUM_MC_SIM+2), np.ones((MAX_NUM_MC_SIM+1))*ICTHRESH,'r--',label='IC threshold',lw=2)
    plt.title(f'Informative content (IC) for each simulation. Actually conducted analyses : {MAX_NUM_MC_SIM+1}',fontweight='bold')
    plt.xlabel('Simulation number')
    plt.ylabel('IC [-]')
    plt.ylim(0,1)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(RESULTS_PATH + "/IC_sim.png", dpi=300)
    plt.close()


