# Paths definitions
DATA_FILE = 'DATA/5DOF_fixed_ex.txt'
RESULTS_PATH = 'RESULTS'
N_DIM = 2 # Problem dimensions, accepted integers 2 or 3 for plotting mode shapes in 2D or 3D structures
NODESFILE = 'DATA/nodes.txt'
CONNECTIVITYFILE = 'DATA/connectivity.txt'

# Vibration data info
## [Hz]
SAMPLING_FREQUENCY = 100
DECIMATION_FACTOR = 5
FUNDAMENTAL_FREQUENCY = 1.0

# Monte Carlo settings:
MAX_NUM_MC_SIM_PHASE_1 = 5
MAX_NUM_MC_SIM_PHASE_2 = 16

# Mode shape normalization
# accepted: integer of specific dof otherwise a string 'max'
NORMALIZDOF = 4 #'max'

# KDE prominence peaks
# Prominence may be an arbitrary float between 0 and 1 (suggested 0.1) 
# otherwise the string 'automatic'
KDEPROMINENCE = 'automatic' # 0.1

# RF intelligent core information content IC threshold
ICTHRESH = 0.10

# i-AOMA PHASE 2
BATCHNSIM = 4          # evaluate the convergence every 50 analyses
CONVMCTHRESH = 0.02    # track relative differences covariance matrix within +-2% for
                       # acceptable shifting convergence band rule (ASCBR) 

# Plotting flags
PLOTSIGNALS = True
PLOTSVDOFPSD = True
SAVEPLOTSIGNALS = True
SAVEPLOTSVDOFPSD = True

PLOT_OVERLAPPED_SD = True

# Plotting options
# Standard deviation factor
MODESCALEFCT = 1
MODESTDFCT = 3
MODESTDFCT_ELLIPSES = 3 # it is accounted only for 3D problems
MODE2D_DIRECTION = 'vertical' # it is accounted only for 2D problems, accepted 'vertical' or 'horizontal'

