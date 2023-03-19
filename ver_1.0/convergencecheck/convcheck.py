import numpy as np
from convergencecheck.helpers import *
from plotting_fns.plot_mode_shapes import *

class ConvCheck():
    def __init__(self, Frequency_dataset, MAX_NUM_MC_SIM_PHASE_1):
        # Frequency_dataset is a list with length equal to the number of founded modes, each containing arrays
        # [ ['Frequency', 'Order', 'Label', 'Damp', 'Emme', 'ModeNum', 'SimNumber'], ['SimNumber','dof','dof','...'] ]
        self.MAX_NUM_MC_SIM_PHASE_1 = MAX_NUM_MC_SIM_PHASE_1
        self.numdofs = Frequency_dataset[0][0,8:].shape[0]
        self.frequency_num_clusters = len(Frequency_dataset)
        self.modes_mean = np.zeros( (self.MAX_NUM_MC_SIM_PHASE_1, self.frequency_num_clusters, self.numdofs) )
        self.modes_std = np.zeros( (self.MAX_NUM_MC_SIM_PHASE_1, self.frequency_num_clusters, self.numdofs) )
        self.modes_trace_covariance = np.zeros( (self.MAX_NUM_MC_SIM_PHASE_1, self.frequency_num_clusters) )
        self.modes_trace_covariance_rel_diff = np.zeros( (MAX_NUM_MC_SIM_PHASE_1, self.frequency_num_clusters) )

        self.freq_mean = np.zeros( (self.MAX_NUM_MC_SIM_PHASE_1, self.frequency_num_clusters) )
        self.freq_std = np.zeros( (self.MAX_NUM_MC_SIM_PHASE_1, self.frequency_num_clusters) )

        self.damp_mean = np.zeros( (self.MAX_NUM_MC_SIM_PHASE_1, self.frequency_num_clusters) )
        self.damp_std = np.zeros( (self.MAX_NUM_MC_SIM_PHASE_1, self.frequency_num_clusters) )
        
        for pp in range(len(Frequency_dataset)): # pp cycle for each founded mode clusters
            for jj in range(self.MAX_NUM_MC_SIM_PHASE_1): # jj cycle for each simulation 0, 1, ... , count_sim_effective
                tmp = Frequency_dataset[pp][ Frequency_dataset[pp][:,6] <= jj]
                self.modes_mean[jj, pp, :] = np.mean(tmp[:,8:], axis=0)
                self.modes_std[jj, pp, :] = np.std(tmp[:,8:], axis=0)
                self.modes_trace_covariance[jj, pp] = np.trace( np.cov( tmp[:,8:].T ) )
                # Evaluate also frequency and damping changes during simulations
                self.freq_mean[jj, pp] = np.mean(tmp[:,0])
                self.freq_std[jj, pp] = np.std(tmp[:,0])
                self.damp_mean[jj, pp] = np.mean(tmp[:,3])
                self.damp_std[jj, pp] = np.std(tmp[:,3])

        self.modes_trace_covariance_rel_diff = np.divide( np.diff(self.modes_trace_covariance, axis=0) , abs(self.modes_trace_covariance[:-1]) )

        # helper command: np.savetxt('A.txt', Frequency_dataset[pp][Frequency_dataset[pp][:,6].argsort()], fmt='%.4f') 

    def update_converg_sim(self, Frequency_dataset, count_sim_effective, last_check_sim, CONVMCTHRESH):
        convergence_reached = 0
        tmp_count = count_sim_effective - last_check_sim
        tmp_modes_mean = np.zeros( (tmp_count, self.frequency_num_clusters, self.numdofs) )
        tmp_modes_std = np.zeros( (tmp_count, self.frequency_num_clusters, self.numdofs) )
        tmp_modes_trace_covariance = np.zeros( (tmp_count, self.frequency_num_clusters) )

        tmp_freq_mean = np.zeros( (tmp_count, self.frequency_num_clusters) )
        tmp_freq_std = np.zeros( (tmp_count, self.frequency_num_clusters) )

        tmp_damp_mean = np.zeros( (tmp_count, self.frequency_num_clusters) )
        tmp_damp_std = np.zeros( (tmp_count, self.frequency_num_clusters) )


        for pp in range(len(Frequency_dataset)): # pp cycle for each founded mode clusters
            for jj, Nsim in enumerate(np.arange(last_check_sim+1, count_sim_effective+1)): # jj is index 0, 1, ... , whereas Nsim is the actual number of sim in that batch
                tmp = Frequency_dataset[pp][ Frequency_dataset[pp][:,6] <= Nsim]
                tmp_modes_mean[jj, pp, :] = np.mean(tmp[:,8:], axis=0)
                tmp_modes_std[jj, pp, :] = np.std(tmp[:,8:], axis=0)
                tmp_modes_trace_covariance[jj, pp] = np.trace( np.cov( tmp[:,8:].T ) )
                # Evaluate also frequency and damping changes during simulations
                tmp_freq_mean[jj, pp] = np.mean(tmp[:,0])
                tmp_freq_std[jj, pp] = np.std(tmp[:,0])
                tmp_damp_mean[jj, pp] = np.mean(tmp[:,3])
                tmp_damp_std[jj, pp] = np.std(tmp[:,3])
        tmp_modes_trace_covariance_rel_diff = np.divide( np.diff(np.vstack( (self.modes_trace_covariance[-1], tmp_modes_trace_covariance) ) ,axis=0) , \
                                                         abs(tmp_modes_trace_covariance) )

        # if (abs(tmp_modes_trace_covariance_rel_diff) <= CONVMCTHRESH).all() :
        if (np.nan_to_num(abs(tmp_modes_trace_covariance_rel_diff), copy=True, nan=0.0, posinf=None, neginf=None) <= CONVMCTHRESH).all() :
                convergence_reached = 1
        
        # updates values
        self.modes_trace_covariance_rel_diff = np.vstack( (self.modes_trace_covariance_rel_diff, tmp_modes_trace_covariance_rel_diff) )

        self.modes_mean = np.vstack( (self.modes_mean, tmp_modes_mean) )
        self.modes_std = np.vstack( (self.modes_std, tmp_modes_std) )
        self.modes_trace_covariance = np.vstack( (self.modes_trace_covariance, tmp_modes_trace_covariance) )

        self.freq_mean = np.vstack( (self.freq_mean, tmp_freq_mean) )
        self.freq_std = np.vstack( (self.freq_std, tmp_freq_std) )
        self.damp_mean = np.vstack( (self.damp_mean, tmp_damp_mean) )
        self.damp_std = np.vstack( (self.damp_std, tmp_damp_std) )

        return convergence_reached

    def export_results_to_file(self, RESULTS_PATH):
        with open(RESULTS_PATH+'/modes_mean.npy', 'wb') as f:
                np.save(f, self.modes_mean)
        with open(RESULTS_PATH+'/modes_std.npy', 'wb') as f:
                np.save(f, self.modes_std)
        with open(RESULTS_PATH+'/modes_trace_covariance.npy', 'wb') as f:
                np.save(f, self.modes_trace_covariance)
        with open(RESULTS_PATH+'/modes_trace_covariance_rel_diff.npy', 'wb') as f:
                np.save(f, self.modes_trace_covariance_rel_diff)
        with open(RESULTS_PATH+'/freq_mean.npy', 'wb') as f:
                np.save(f, self.freq_mean)
        with open(RESULTS_PATH+'/freq_std.npy', 'wb') as f:
                np.save(f, self.freq_std)
        with open(RESULTS_PATH+'/damp_mean.npy', 'wb') as f:
                np.save(f, self.damp_mean)
        with open(RESULTS_PATH+'/damp_std.npy', 'wb') as f:
                np.save(f, self.damp_std)

    def plot_mode_shapes(self, N_DIM, nodes, connectivity, connectivity_mode_shape_dofs, MODESCALEFCT, MODESTDFCT, \
                         MODESTDFCT_ELLIPSES, MODE_SHAPE_PAPER, RESULTS_PATH):
        if N_DIM == 2:
                if MODE_SHAPE_PAPER==None:
                        plot_modes_for_2d(self.modes_mean[-1], self.modes_std[-1], self.freq_mean[-1], nodes, \
                                                connectivity, connectivity_mode_shape_dofs, MODESCALEFCT, MODESTDFCT, MODE_SHAPE_PAPER, RESULTS_PATH)
                else:
                        plot_modes_for_2d_for_paper(self.modes_mean[-1], self.modes_std[-1], self.freq_mean[-1], nodes, \
                                                        connectivity, connectivity_mode_shape_dofs, MODESCALEFCT, MODESTDFCT, MODE_SHAPE_PAPER, RESULTS_PATH)
                 
        elif N_DIM == 3:
                if MODE_SHAPE_PAPER==None:
                        plot_modes_for_3d(self.modes_mean[-1], self.modes_std[-1], self.freq_mean[-1], nodes, \
                                                connectivity, connectivity_mode_shape_dofs, MODESCALEFCT, RESULTS_PATH)
                else:
                        plot_modes_for_3d_for_paper(self.modes_mean[-1], self.modes_std[-1], self.freq_mean[-1], nodes, \
                                        connectivity, connectivity_mode_shape_dofs, MODESCALEFCT, MODESTDFCT, RESULTS_PATH)
        else:
            raise ValueError("Unrecognized Number of dimensions to plot mode shapes")


    def plot_convergence_curves(self,CONVMCTHRESH, RESULTS_PATH ):
        plot_trace(self.modes_trace_covariance, self.modes_trace_covariance_rel_diff, self.freq_mean[-1], CONVMCTHRESH, RESULTS_PATH)
        # plot_cov_modes() # TBC
        plot_frequency_conv(self.freq_mean, self.freq_std, RESULTS_PATH)
        plot_damp_conv(self.damp_mean, self.damp_std, self.freq_mean[-1], RESULTS_PATH)


