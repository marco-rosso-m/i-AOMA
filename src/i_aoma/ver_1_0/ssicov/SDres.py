from ssicov.helpers import *
import ssicov.constants as c


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,AutoMinorLocator

class SdRes():

    def __init__(self, ssicovSD, admissiblepar):
        # Save the normalized qmc halton sampled parameters and vstack new effective parameters [time shift, max order, window lenght, time target]
        self.admissiblepar = admissiblepar
        # Save total number of stable poles in stab diag and append new number for new effective parameters to the list
        self.selectedpoles_totnum = [ssicovSD.selectedpoles_totnum]
        # Save selected stable poles and corresponding mode shapes
        self.selectedpoles = ssicovSD.selectedpoles
        self.selectedpoles_col_names = ssicovSD.selectedpoles_col_names
        # column names ['Frequency', 'Order', 'Label', 'Damp', 'Emme', 'ModeNum', 'SimNumber']
        self.selectedmodes = ssicovSD.selectedmodes

        # Save all the poles of each k-th simulation to show totally overlapped graphs
        self.ReducedPoles_col_names = ssicovSD.ReducedPoles_col_names
        # column names ['Frequency', 'Order', 'Label']
        self.ReducedPoles = ssicovSD.ReducedPoles

    def update_results(self, ssicovSD, admissiblepar):
        # Update normalized qmc halton sampled parameters with vstack new effective parameters [time shift, max order, window lenght, time target]
        self.admissiblepar = np.vstack( (self.admissiblepar, admissiblepar) )
        # Update total number of stable poles in stab diag and append new number for new effective parameters to the list
        self.selectedpoles_totnum.append(ssicovSD.selectedpoles_totnum)
        # Update selected stable poles and corresponding mode shapes
        self.selectedpoles = np.vstack( (self.selectedpoles, ssicovSD.selectedpoles) )
        self.selectedmodes = np.vstack( (self.selectedmodes, ssicovSD.selectedmodes) )

        # Update all the poles of each k-th simulation to show totally overlapped graphs
        self.ReducedPoles = np.vstack( (self.ReducedPoles, ssicovSD.ReducedPoles) )

    def plot_overlapped_SD(self,fs,PLOT_OVERLAPPED_SD,RESULTS_PATH):
        if PLOT_OVERLAPPED_SD:

            _colors = {0:'Red', 1:'darkorange', 2:'gold', 3:'yellow', 4:'Green'}
            fig1, ax1 = plt.subplots(figsize=(10,5),facecolor='white')
            ax1 = sns.scatterplot(x=self.ReducedPoles[:,0], y=self.ReducedPoles[:,1]*2, hue=self.ReducedPoles[:,2], palette=_colors, legend=False)
            ax1 = sns.scatterplot(x=self.selectedpoles[:,0], y=self.selectedpoles[:,1]*2, color='None',s=40,linewidth=0.1, edgecolor="#023e7d",legend=False)

            plt.xlim(left=0, right=fs/2)
            plt.ylim(bottom=0, top=max(self.ReducedPoles[:,1]*2))
            ax1.set_title(f'Overlapped Stabilization Diagrams and Selecting Stable Poles',fontweight='bold')
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('Orders')
            legend_elements = [Line2D([0], [0], marker='o', color=_colors[0], lw=0, label='0'),
                            Line2D([0], [0], marker='o', color=_colors[1], lw=0, label='1'),
                            Line2D([0], [0], marker='o', color=_colors[2], lw=0, label='2'),
                            Line2D([0], [0], marker='o', color=_colors[3], lw=0, label='3'),
                            Line2D([0], [0], marker='o', color=_colors[4], lw=0, label='4'),
                            Line2D([0], [0], marker='o', color='w', lw=0, markeredgecolor='#023e7d', markerfacecolor=None, label='Selected')]
            ax1.legend(handles=legend_elements, loc='lower right',framealpha=0.95,title="Labels:")
            plt.savefig(RESULTS_PATH+f"/Overlapped_SD.png", dpi=150)
            plt.savefig(RESULTS_PATH+f"/Overlapped_SD.pdf")
            # plt.show()
            plt.close()

    def plot_overlapped_SD_stable(self,fs,PLOT_OVERLAPPED_SD,RESULTS_PATH):
        if PLOT_OVERLAPPED_SD:

            _colors = {0:'Red', 1:'darkorange', 2:'gold', 3:'yellow', 4:'Green'}
            fig1, ax1 = plt.subplots(figsize=(10,5),facecolor='white')
            ax1 = sns.scatterplot(x=self.selectedpoles[:,0], y=self.selectedpoles[:,1]*2, hue=self.selectedpoles[:,2], palette=_colors, legend=False)

            plt.xlim(left=0, right=fs/2)
            plt.ylim(bottom=0, top=max(self.ReducedPoles[:,1]*2))
            ax1.set_title(f'Overlapped Stabilization Diagrams with Stable Poles only',fontweight='bold')
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('Orders')
            legend_elements = [Line2D([0], [0], marker='o', color=_colors[4], lw=0, label='4')]
            ax1.legend(handles=legend_elements, loc='lower right',framealpha=0.95,title="Labels:")
            plt.savefig(RESULTS_PATH+f"/Overlapped_SD_stables.png", dpi=150)
            plt.savefig(RESULTS_PATH+f"/Overlapped_SD_stables.pdf")
            # plt.show()
            plt.close()

    # export results to npz file
    def export_results_to_file(self,RESULTS_PATH):

        with open(RESULTS_PATH+'/selectedpoles.npy', 'wb') as f:
                np.save(f, self.selectedpoles)

        with open(RESULTS_PATH+'/selectedmodes.npy', 'wb') as f:
                np.save(f, self.selectedmodes)

        with open(RESULTS_PATH+'/admissiblepar.npy', 'wb') as f:
                np.save(f, self.admissiblepar)

        with open(RESULTS_PATH+'/selectedpoles_totnum.npy', 'wb') as f:
                np.save(f, self.selectedpoles_totnum)

        with open(RESULTS_PATH+'/ReducedPoles.npy', 'wb') as f:
                np.save(f, self.ReducedPoles)
        # Test loading file
        # X = np.load(RESULTS_PATH+'/Phase1'+'/selectedpoles.npy', allow_pickle = True)

    def jointpolesarray_and_modesarray(self):
        #   [ ['Frequency', 'Order', 'Label', 'Damp', 'Emme', 'ModeNum', 'SimNumber'], ['SimNumber','dof','dof','...'] ]
        self.joint_col_names = [self.selectedpoles_col_names, ['SimNumber','dof','dof','...']]
        self.jointpolesmodes = np.hstack( (self.selectedpoles, self.selectedmodes) )
        return self.jointpolesmodes

    def plot_sampling_maps(self, RESULTS_PATH):
        plot_sampled_maps(self.admissiblepar, RESULTS_PATH)
