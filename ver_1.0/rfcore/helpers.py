import numpy as np


def joint_admissiblepar_discardedpar(admissiblepar, discardedpar, y):
    x = np.vstack( (admissiblepar, discardedpar) )
    y = np.hstack( (y , np.zeros(discardedpar.shape[0]) ) )
    return x, y



