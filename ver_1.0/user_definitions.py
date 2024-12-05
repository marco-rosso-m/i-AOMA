# Paths definitions
DATA_FILE = 'DATA/data_tower.pkl' #'DATA/data_tower.pkl' #'DATA/TRAVE1_AF_1cuscino.csv' #'DATA/5DOF_fixed_ex.txt'
RESULTS_PATH = 'RESULTS/tower' #'RESULTS/tower'
NODESFILE = 'DATA/nodes_tower.xlsx' #'DATA/nodes_tower.xlsx' #'DATA/nodes_wood.txt' #'DATA/nodes.txt'
CONNECTIVITYFILE = 'DATA/connectivity_tower.xlsx' #'DATA/connectivity_tower.xlsx' #'DATA/connectivity_wood.txt' #'DATA/connectivity.txt'
# The following file containts 0 and 1 to attach mode shape component to corresponding DOFs of the geometry
MONITORED_DOF_FILE = 'DATA/nodes_mode_shape_dofs_tower.xlsx' #'DATA/nodes_mode_shape_dofs_tower.xlsx' # 'DATA/nodes_mode_shape_dofs.txt'
# Problem dimensions, accepted integers 2 or 3 for plotting mode shapes in 2D or 3D structures
N_DIM = 3

# Vibration data info
## [Hz]
SAMPLING_FREQUENCY = 200 #200 #1200 #100
DECIMATION_FACTOR = 40 #40 #2 #5
FUNDAMENTAL_FREQUENCY = 1.0 #66.07 #1.0
DATAFILTERING = False
# butterworth filter design parameters [N,Wn,btype] N:order of the filter; Wn: cut-off frequency; btype: ‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’
BUTTERWORTH_DESIGN = [5,30,'highpass'] 

# Monte Carlo settings:
MAX_NUM_MC_SIM_PHASE_1 = 5
MAX_NUM_MC_SIM_PHASE_2 = 4

# Mode shape normalization
# accepted: integer of specific dof otherwise a string 'max'
# in this case dof numbering follows the data file columns. E.g. 0 stands for the column 0 of the data file
NORMALIZDOF = 0 #'max' #0 #0 #4

# KDE prominence peaks
# Prominence may be an arbitrary float between 0 and 1 (suggested 0.1) 
# otherwise the string 'automatic'
KDEPROMINENCE = 'automatic' # 0.1
LEGEND_KWARGS_PLOTFREQCLUSTERS = {'loc': 'upper right','ncols': 2, 'fontsize': 10}

# RF intelligent core information content IC threshold
ICTHRESH = 0.10

# i-AOMA PHASE 2
BATCHNSIM = 4         # evaluate the convergence every 50 analyses
CONVMCTHRESH = 0.02    # track relative differences covariance matrix within +-2% for
                       # acceptable shifting convergence band rule (ASCBR) 

# Plotting flags
PLOTSIGNALS = True
LEGEND_KWARGS_PLOTSIGNALS = {'loc': 'best','ncols': 6, 'fontsize': 5.9}
PLOTSVDOFPSD = True
LEGEND_KWARGS_PLOTSVD = {'loc': 'lower left','ncols': 6, 'fontsize': 5.9}
SAVEPLOTSIGNALS = True
SAVEPLOTSVDOFPSD = True

PLOT_OVERLAPPED_SD = True

# Plotting options
# Scale factors
MODESCALEFCT = 10 #1

MODESTDFCT = 1

MODE_SHAPE_PAPER = 'vertical'  #'vertical' # for 2D problems, accepted 'vertical' or 'horizontal' , normally set to None for general plot

MODESTDFCT_ELLIPSES = 6 # it is accounted only for 3D problems
