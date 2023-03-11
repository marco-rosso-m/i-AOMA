#utilities functions
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
# from PyOMA import FDDsvp
from scipy import signal as signalscipy
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
import mplcursors


def create_results_folder(RESULTS_PATH):
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH) 

def import_data(DATA_FILE: str):
    _, file_extension = os.path.splitext(DATA_FILE)

    if file_extension == '.csv' or file_extension == '.txt':
        data = pd.read_csv(DATA_FILE, header=None, sep="\s+", index_col=False) 
        data = data.to_numpy()
    elif file_extension == '.pkl':
        with open(DATA_FILE, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"File extension not recognized: {file_extension} : Supported file format .txt, .csv, .pkl")
    return data

def visualize_plot_signals(data, PLOTSIGNALS, RESULTS_PATH, SAMPLING_FREQUENCY, SAVEPLOTSIGNALS, labelsformat):
    fs=SAMPLING_FREQUENCY
    time_units, vibration_units=labelsformat[0],labelsformat[1]
    if PLOTSIGNALS:
        plt.figure()
        for ii in range(data.shape[1]):
            plt.plot(np.arange(start=0,stop=data.shape[0]/fs,step=1/fs),data[:,ii],label=f'channel {ii+1}')
        plt.legend(loc='best');plt.xlabel(time_units);plt.ylabel(vibration_units)
        plt.title('Monitored signals')
        plt.tight_layout()
        if SAVEPLOTSIGNALS:
            plt.savefig(RESULTS_PATH+f'/Monitored_Signals.png', format='png')
        # plt.show()
        plt.close()

# Function taken from Py-OMA module
def FDDsvp(data, fs, df=0.01, pov=0.5, window='hann'):
    '''
    This function perform the Frequency Domain Decomposition algorithm.
    
    The function return the plot of the singular values of the power spectral
    density. The cross power spectral density is estimated using 
    scipy.signal.csd() function, which in turn is based on Welch's method.
    Furthermore it returns a dictionary that contains the results needed
    by the function FDDmodEX().
    ----------
    Parameters
    ----------
    data : array
        The time history records (Ndata x Nchannels).
    fs : float
        The sampling frequency.
    df : float
        Desired frequency resolution. Default to 0.01 (Hz).
    pov : float
        Percentage of overlap between segments. Default to 50%.
    window : str or tuple or array_like
        Desired window to use. Window is passed to scipy.signal's get_window
        function (see SciPy.org for more info). Default to "hann" which stands
        for a “Hanning” window.

    -------
    Returns
    -------
    fig1 : matplotlib figure
        Plot of the singular values of the power spectral matrix.
    Results : dictionary
        Dictionary of results to be passed to FDDmodEX()
    '''  
    
    # ndat=data.shape[0] # Number of data points
    nch=data.shape[1] # Number of channels
    freq_max = fs/2 # Nyquist frequency
    nxseg = fs/df # number of point per segments
#    nseg = ndat // nxseg # number of segments
    noverlap = nxseg // (1/pov) # Number of overlapping points
    
    # Initialization
    PSD_matr = np.zeros((nch, nch, int((nxseg)/2+1)), dtype=complex) 
    S_val = np.zeros((nch, nch, int((nxseg)/2+1))) 
    S_vec = np.zeros((nch, nch, int((nxseg)/2+1)), dtype=complex) 
    
    # Calculating Auto e Cross-Spectral Density
    for _i in range(0, nch):
        for _j in range(0, nch):
            _f, _Pxy = signalscipy.csd(data[:, _i],data[:, _j], fs=fs, nperseg=nxseg, noverlap=noverlap, window=window)
            PSD_matr[_i, _j, :] = _Pxy
            
    # Singular value decomposition     
    for _i in range(np.shape(PSD_matr)[2]):
        U1, S1, _V1_t = np.linalg.svd(PSD_matr[:,:,_i])
        U1_1=np.transpose(U1) 
        S1 = np.diag(S1)
        S_val[:,:,_i] = S1
        S_vec[:,:,_i] = U1_1
    
    # Plot of the singular values in log scale
    fig, ax = plt.subplots()
    for _i in range(nch):
    #    ax.semilogy(_f, S_val[_i, _i]) # scala log
        ax.plot(_f[:], 10*np.log10(S_val[_i, _i])) # decibel
    ax.grid()
    ax.set_xlim(left=0, right=freq_max)
    ax.xaxis.set_major_locator(MultipleLocator(freq_max/10))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.xaxis.set_minor_locator(MultipleLocator(freq_max/100))
    ax.set_title("Singular values plot - (Freq. res. ={0})".format(df))
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'dB $[g^2/Hz]$')    
    # ax.set_ylabel(r'dB $\left[\frac{\left(\frac{m}{s^2}\right)^2}{Hz}\right]$')    
    mplcursors.cursor()
    
    Results={}
    Results['Data'] = {'Data': data}
    Results['Data']['Samp. Freq.'] = fs
    Results['Data']['Freq. Resol.'] = df
    Results['Singular Values'] = S_val
    Results['Singular Vectors'] = S_vec
    Results['PSD Matrix'] = PSD_matr
    
    return fig, Results

def visualize_plot_PSD(data, PLOTSVDOFPSD, RESULTS_PATH, SAMPLING_FREQUENCY, SAVEPLOTSVDOFPSD):
    fs=SAMPLING_FREQUENCY
    if PLOTSVDOFPSD:
        FDD = FDDsvp(data,  fs)
        if SAVEPLOTSVDOFPSD:
            plt.savefig(RESULTS_PATH+f'/SVD_of_PSD.png', format='png')
        # plt.show()
        plt.close()
    return FDD