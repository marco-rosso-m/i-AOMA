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