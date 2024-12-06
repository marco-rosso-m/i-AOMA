# helpers functions
from scipy import signal as sig




def dataslice(data, winlength, targetInstant):

    if targetInstant-int(winlength/2)<0:
        Sliceddata = data[0:int(winlength),:]
    elif targetInstant+int(winlength/2)>data.shape[0]:
        Sliceddata = data[-int(winlength):,:]
    else:
        Sliceddata = data[targetInstant-int(winlength/2):targetInstant+int(winlength/2),:]

    return Sliceddata

def datadecimate(data, fs, q=1, ftype='fir', axis=0):
    if q>1:
        data = sig.decimate(data,  q, ftype='fir', axis=0) # Decimation
    fs = fs/q # [Hz] Decimated sampling frequency
    return data,fs

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sig.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def datafiltering(data, fs, BUTTERWORTH_DESIGN):
    # butter_highpass_filter
    # BUTTERWORTH_DESIGN = [N,Wn,btype]
    b, a = butter_highpass(cutoff=BUTTERWORTH_DESIGN[1], fs=fs, order=BUTTERWORTH_DESIGN[0])
    y = sig.filtfilt(b, a, data, axis=0)
    return y
