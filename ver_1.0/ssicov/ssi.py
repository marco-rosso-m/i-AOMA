from ssicov.helpers import *
import ssicov.constants as c


import sys

class SsiCov():

    def __init__(self, data, fs, ts, ordmax):
        self.data = data
        self.fs = fs
        self.ts = ts
        self.ordmax = ordmax

        results = SSIcovStaDiag(self.data, self.fs, self.ts, self.ordmax)
        # results.keys() = dict_keys(['Data', 'All Poles', 'Reduced Poles', 'Modes'])
        # results is a dictionary containing: 'Data' array; 'All Poles' dataframe; 'Reduced Poles' dataframe; 'Modes' array;  
        # self.Data = results['Data']
        # self.AllPoles = results['All Poles']
        self.ReducedPoles = results['Reduced Poles']
        self.Modes = results["Modes"]

        # self.AllPoles = self.AllPoles.sort_values(['Order','Frequency'])
        self.ReducedPoles = self.ReducedPoles.sort_values(['Order','Frequency'])

    def mode_normalization(self,dof='max'):
        if dof == "max": # mode normalization with respect to the dof which presents the maximum value of the mode shape
            for oo in range(len(self.Modes)):
                for jj in range(self.Modes[oo].shape[1]):
                    self.Modes[oo][:,jj]=self.Modes[oo][:,jj].real
                    tmp_max=max(self.Modes[oo][:,jj],key=abs)
                    self.Modes[oo][:,jj]=self.Modes[oo][:,jj]/tmp_max
        else:
            try:
                for oo in range(len(self.Modes)):
                    for jj in range(self.Modes[oo].shape[1]):
                        self.Modes[oo][:,jj]=self.Modes[oo][:,jj].real 
                        tmp_max=self.Modes[oo][:,jj][dof]  # dof is an integer referring to the DOF with respect to normalize
                        self.Modes[oo][:,jj]=self.Modes[oo][:,jj]/tmp_max
            except:
                print('Mode Normalization Error: Normalization allowed or for "max" value or expliciting the DOF to normalize as integer')
                sys.exit(1)

    def retain_stable_poles(self, sim_num):
        # select only stable poles from stabilization diagram within the shannon-nyquist frequency
        self.selectedpoles = self.ReducedPoles[self.ReducedPoles['Label'] >= c.STABLEPOLESID]
        self.selectedpoles = self.selectedpoles[self.selectedpoles['Frequency'] <= self.fs/2]
        # add two more column to denote the number of the index of mode array and the current simulation number
        self.selectedpoles['ModeNum'] = np.array(range(self.selectedpoles.shape[0]))
        self.selectedpoles['SimNumber'] = np.ones(self.selectedpoles.shape[0], dtype=int) * sim_num

        tmpModeID = self.selectedpoles ['Emme'].tolist()
        tmpOrder = self.selectedpoles ['Order'].tolist()
        # total number of stable poles in the stab diagram
        self.selectedpoles_totnum = self.selectedpoles.shape[0]

        self.selectedmodes = np.zeros((self.selectedpoles.shape[0], self.Modes[1][:,0].shape[0] ))

        for kk in range(self.selectedpoles.shape[0]):
            self.selectedmodes[kk,:]= np.real(self.Modes[tmpOrder[kk]][:,tmpModeID[kk]])

        # first column of mode shape denote the simulation number
        self.selectedmodes = np.hstack((self.selectedpoles['SimNumber'].to_numpy().reshape(-1,1),self.selectedmodes))
        # get name of each column
        self.selectedpoles_col_names = list(self.selectedpoles.columns)
        # convert dataframe in numpy to be lighter and faster
        self.selectedpoles = self.selectedpoles.to_numpy()
        # store the columns of order, frequency and label as numpy array to plot overlapped poles
        self.ReducedPoles_col_names = list(self.ReducedPoles.iloc[:,:3].columns)
        self.ReducedPoles = self.ReducedPoles.iloc[:,:3].to_numpy()




