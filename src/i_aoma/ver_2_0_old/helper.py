import os
import signal
import functools
import errno
import numpy as np

# import time
# import numpy as np
# from multiprocessing import shared_memory
import matplotlib
import matplotlib.pyplot as plt

from memory_profiler import profile

matplotlib.use("agg")


class TimeoutError(Exception):
    pass


def timeout(seconds=30, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError()  # TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator


def run_SSICov(SingleSetup, qmc_sample, qmc_sample_unitary, NumAnal):
    """
    This function runs the SSICov analysis using the SingleSetup object and the qmc_sample.
    Reminders:
        HaltonSamples array of shape (_numsim, _dim) is generated using the Halton sequence.
        HaltonSamples[:, 0] corresponds to the timeshift parameter.
        HaltonSamples[:, 1] corresponds to the order parameter.
        HaltonSamples[:, 2] corresponds to the window_length parameter.
        HaltonSamples[:, 3] corresponds to the time_target_centering_window parameter.

        Scrambling can help distribute the points more uniformly across the space, reducing clustering and gaps.
    """
    print("SSICov analysis...")

    # # Simulate a busy task by performing computations
    # start_time = time.time()
    # while time.time() - start_time < 10:
    #     # Perform some dummy computations to simulate a busy task
    #     result = sum(i * i for i in range(10000))
    # print("SSICov analysis completed.")

    # Check if SingleSetup is using shared memory
    # if isinstance(SingleSetup, shared_memory.SharedMemory):
    #     print("SingleSetup is using shared memory.")
    # else:
    #     print("SingleSetup is not using shared memory.")

    # print(SingleSetup.data.shape,SingleSetup.algorithms['SSIcov'].run_params.br,SingleSetup.algorithms['SSIcov'].run_params.ordmax)

    SingleSetup.run_by_name("SSIcov")

    fig, ax = SingleSetup.algorithms["SSIcov"].plot_stab(
        freqlim=(0, SingleSetup.fs / 2), hide_poles=False
    )
    ax.set_title(
        f"Sim. {NumAnal}, br={qmc_sample[0][0]}, ordmax={qmc_sample[0][1]}, wlen={qmc_sample_unitary[0][2]*100:.1f}$\%$, tt={qmc_sample_unitary[0][3]*100:.1f}$\%$"
    )
    plt.savefig(
        f"src/i_aoma/ver_2_0/Results/trave1_results_1cuscino/{NumAnal+1000}_StabDiag.png",
        dpi=300,
    )
    plt.close()

    fig, ax = SingleSetup.algorithms["SSIcov"].plot_cluster(
        freqlim=(0, SingleSetup.fs / 2), hide_poles=False
    )
    ax.set_title(
        f"Sim. {NumAnal}, br={qmc_sample[0][0]}, ordmax={qmc_sample[0][1]}, wlen={qmc_sample_unitary[0][2]*100:.1f}$\%$, tt={qmc_sample_unitary[0][3]*100:.1f}$\%$"
    )
    plt.savefig(
        f"src/i_aoma/ver_2_0/Results/trave1_results_1cuscino/{NumAnal+1000}_DampingClusters.png",
        dpi=300,
    )
    plt.close()

    return SingleSetup.algorithms["SSIcov"].result


def MaC(Fi1, Fi2):
    """
    This function returns the Modal Assurance Criterion (MAC) for two mode
    shape vectors.

    If the input arrays are in the form (n,) (1D arrays) the output is a
    scalar, if the input are in the form (n,m) the output is a (m,m) matrix
    (MAC matrix).

    ----------
    Parameters
    ----------
    Fi1 : array (1D or 2D)
        First mode shape vector (or matrix).
    Fi2 : array (1D or 2D)
        Second mode shape vector (or matrix).

    -------
    Returns
    -------
    MAC : float or (2D array)
        Modal Assurance Criterion.
    """

    MAC = np.abs(Fi1.conj().T @ Fi2) ** 2 / (
        (Fi1.conj().T @ Fi1) * (Fi2.conj().T @ Fi2)
    )

    return MAC


@profile(
    precision=4, stream=open("src/i_aoma/ver_2_0/memprof_run_SSICov_pyoma1.log", "a+")
)
def run_SSICov_pyoma1(
    SingleSetup,
    qmc_sample,
    qmc_sample_unitary,
    NumAnal,  # ):
    lim=(0.01, 0.05, 0.02, 0.1),
    method="1",
):  # def SSIcovStaDiag(data, fs, br, ordmax=None, lim=(0.01,0.05,0.02,0.1), method='1'):
    """
    This function perform the covariance-driven Stochastic sub-Space
    Identification algorithm.

    The function returns the Stabilization Diagram (Plot) for the given
    data. Furthermore it returns a dictionary that contains the results needed
    by the function SSImodEX().

    ----------
    Parameters
    ----------
    data : 2D array
        The time history records (N°data points x N°channels).
    fs : float
        The sampling frequency.
    br : integer
        The number of block rows (time shifts).
    ordmax : None or integer
        The maximum model order to use in the construction of the
        stabilisation diagram. None (default) is equivalent to the maximum
        allowable model order equal to br*data.shape[1].
    lim : tuple
        Limit values to use for the stability requirements of the poles. The
        first three values are used to check the stability of the poles.
            - Frequency: (f(n)-f(n+1))/f(n) < lim[0] (default to 0.01)
            - Damping: (xi(n)-xi(n+1))/xi(n) < lim[1] (default to 0.05)
            - Mode shape: 1-MAC((phi(n),phi(n+1)) < lim[2] (default to 0.02)
        The last value (lim[3]) is used to remove all the poles that have a
        higher damping ratio (default to 0.1, N.B. in structural dynamics
        we usually deal with underdamped system)

    method : "1" or "2"
        Method to use in the estimation of the state matrix A:
            - method "1" (default) : the first method takes advantages of the
                                     shift structrure of the observability
                                     matrix.
            - method "2" : the second method is based on the decomposition
                           property of the one-lag shifted Toeplitz matrix.
    -------
    Returns
    -------
    fig1 : matplotlib figure
        Stabilisation diagram.
        Take advantage of the mplcursors module to identify the stable poles.
    Results : dictionary
        Dictionary of results.
        This dictionary will be passed as argument to the SSImodEX() function.
    """
    data = SingleSetup.data
    fs = SingleSetup.fs
    br = qmc_sample[0][0]
    ordmax = qmc_sample[0][1]

    ndat = data.shape[0]  # Number of data points
    nch = data.shape[1]  # Number of channel

    # If the maximum order is not given (default) it is set as the maximum
    # allowable model order which is: number of block rows * number of channels
    if ordmax == None:
        ordmax = br * nch

    freq_max = fs / 2  # Nyquist Frequency

    # unpack the limits used for the construction of the Stab Diag
    lim_f, lim_s, lim_ms, lim_s1 = lim[0], lim[1], lim[2], lim[3]

    Yy = data.T  #

    # =============================================================================
    # Calculating R[i] (with i from 0 to 2*br)
    R_is = np.array(
        [
            1 / (ndat - _s) * (Yy[:, : ndat - _s] @ Yy[:, _s:].T)
            for _s in range(br * 2 + 1)
        ]
    )

    # Assembling the Toepliz matrix
    Tb = np.vstack(
        [
            np.hstack([R_is[_o, :, :] for _o in range(br + _l, _l, -1)])
            for _l in range(br)
        ]
    )

    # One-lag shifted Toeplitz matrix (used in "NExT-ERA" method)
    Tb2 = np.vstack(
        [
            np.hstack([R_is[_o, :, :] for _o in range(br + _l, _l, -1)])
            for _l in range(1, br + 1)
        ]
    )

    # SINGULAR VALUE DECOMPOSITION
    U1, S1, V1_t = np.linalg.svd(Tb)
    S1 = np.diag(S1)
    S1rad = np.sqrt(S1)

    # =============================================================================
    # initializing arrays
    Fr = np.full(
        (ordmax, int((ordmax) / 2 + 1)), np.nan
    )  # initialization of the matrix that contains the frequencies
    Fr_lab = np.full(
        (ordmax, int((ordmax) / 2 + 1)), np.nan
    )  # initialization of the matrix that contains the labels of the poles
    Sm = np.full(
        (ordmax, int((ordmax) / 2 + 1)), np.nan
    )  # initialization of the matrix that contains the damping ratios
    Ms = []  # initialization of the matrix (list of arrays) that contains the mode shapes
    for z in range(0, int((ordmax) / 2 + 1)):
        Ms.append(np.zeros((nch, z * (2))))

    # loop for increasing order of the system
    for _ind in range(0, ordmax + 1, 2):
        S11 = np.zeros((_ind, _ind))  # Inizializzo
        U11 = np.zeros((br * nch, _ind))  # Inizializzo
        V11 = np.zeros((_ind, br * nch))  # Inizializzo
        O_1 = np.zeros((br * nch - nch, _ind))  # Inizializzo
        O_2 = np.zeros((br * nch - nch, _ind))  # Inizializzo

        # Extraction of the submatrices for the increasing order of the system
        S11[:_ind, :_ind] = S1rad[:_ind, :_ind]  #
        U11[: br * nch, :_ind] = U1[: br * nch, :_ind]  #
        V11[:_ind, : br * nch] = V1_t[:_ind, : br * nch]  #

        O = U11 @ S11  # Observability matrix
        # _GAM = S11 @ V11 # Controllability matrix

        O_1[:, :] = O[: O.shape[0] - nch, :]
        O_2[:, :] = O[nch:, :]

        # Estimating matrix A
        if method == "2":
            A = (
                np.linalg.inv(S11) @ U11.T @ Tb2 @ V11.T @ np.linalg.inv(S11)
            )  # Method 2 "NExT-ERA"
        else:
            A = np.linalg.pinv(O_1) @ O_2  # Method 1 (BALANCED_REALIZATION)

        [_AuVal, _AuVett] = np.linalg.eig(A)
        Lambda = (np.log(_AuVal)) * fs
        fr = abs(Lambda) / (2 * np.pi)  # natural frequencies
        smorz = -((np.real(Lambda)) / (abs(Lambda)))  # damping ratios
        # =============================================================================
        # This is a fix for a bug. We make shure that there are not nans
        # (it has, seldom, happened that at the first iteration the first
        # eigenvalue was negative, yielding the log to return a nan that
        # messed up with the plot of the stabilisation diagram)
        for _j in range(len(fr)):
            if np.isnan(fr[_j]) == True:
                fr[_j] = 0
        # =============================================================================
        # Output Influence Matrix
        C = O[:nch, :]

        # Complex mode shapes
        Mcomp = C @ _AuVett
        # Mreal = np.real(C@_AuVett)

        # we are increasing 2 orders at each step
        _ind_new = int(_ind / 2)

        Fr[: len(fr), _ind_new] = fr  # save the frequencies
        Sm[: len(fr), _ind_new] = smorz  # save the damping ratios
        Ms[_ind_new] = Mcomp  # save the mode shapes

        # =============================================================================
        # Check stability of poles
        # 0 = Unstable pole labe
        # 1 = Stable for frequency
        # 2 = Stable for frequency and damping
        # 3 = Stable for frequency and mode shape
        # 4 = Stable pole

        for idx, (_freq, _smor) in enumerate(zip(fr, smorz)):
            if (
                _ind_new == 0 or _ind_new == 1
            ):  # at the first iteration every pole is new
                Fr_lab[: len(fr), _ind_new] = 0  #

            else:
                # Find the index of the pole that minimize the difference with iteration(order) n-1
                ind2 = np.nanargmin(
                    abs(_freq - Fr[:, _ind_new - 1])
                    - min(abs(_freq - Fr[:, _ind_new - 1]))
                )

                Fi_n = Mcomp[:, idx]  # Modal shape iteration n
                Fi_nmeno1 = Ms[int(_ind_new - 1)][:, ind2]  # Modal shape iteration n-1

                # aMAC = np.abs(Fi_n@Fi_nmeno1)**2 / ((Fi_n@Fi_n)*(Fi_nmeno1@Fi_nmeno1)) # autoMAC
                aMAC = MaC(Fi_n, Fi_nmeno1)

                cond1 = abs(_freq - Fr[ind2, _ind_new - 1]) / _freq
                cond2 = abs(_smor - Sm[ind2, _ind_new - 1]) / _smor
                cond3 = 1 - aMAC

                if cond1 < lim_f and cond2 < lim_s and cond3 < lim_ms:
                    Fr_lab[idx, _ind_new] = 4  #

                elif cond1 < lim_f and cond3 < lim_ms:
                    Fr_lab[idx, _ind_new] = 3  #

                elif cond1 < lim_f and cond2 < lim_s:
                    Fr_lab[idx, _ind_new] = 2  #

                elif cond1 < lim_f:
                    Fr_lab[idx, _ind_new] = 1  #
                else:
                    Fr_lab[idx, _ind_new] = 0  # Nuovo polo o polo instabile
    # =============================================================================
    # Stabilisation Diagram
    # =============================================================================
    # Flatten everything
    _x = Fr.flatten(order="f")
    _y = np.array([_i // len(Fr) for _i in range(len(_x))])
    _l = Fr_lab.flatten(order="f")
    _d = Sm.flatten(order="f")
    # Creating a dataframe out of the flattened results


#     df = pd.DataFrame(dict(Frequency=_x, Order=_y, Label=_l, Damp=_d))

# # =============================================================================
#     # Reduced dataframe (without nans) where the modal info is saved
#     df1 = df.copy()
#     df1 = df1.dropna()
#     emme = []
#     # here I look for the index of the shape associated to a given pole
#     for effe,order in zip(df1.Frequency,df1.Order):
#         emme.append(np.nanargmin(abs(effe - Fr[:, order]))) # trovo l'indice
#     # append the list of indexes to the dataframe
#     emme = np.array(emme)
#     df1['Emme'] = emme
# # =============================================================================
#     df2 = df1.copy()
#     # removing the poles that have damping exceding the limit value
#     df2.Frequency = df2.Frequency.where(df2.Damp < lim_s1)
#     # removing the poles that have negative damping
#     df2.Frequency = df2.Frequency.where(df2.Damp > 0)


#     # Physical poles compare in pairs (complex + conjugate)
#     # I look for the poles that DO NOT have a pair and I remove them from the dataframe
#     df3 = df2.Frequency.drop_duplicates(keep=False)
#     df2 = df2.where(~(df2.isin(df3))) #
#     df2 = df2.dropna()# Dropping nans
#     df2 = df2.drop_duplicates(subset='Frequency') # removing conjugates


#     # df4 = df4.where(df2.Order > ordmin).dropna() # Tengo solo i poli sopra ordmin
#     # assigning colours to the labels
#     _colors = {0:'Red', 1:'darkorange', 2:'gold', 3:'yellow', 4:'Green'}

#     fig1, ax1 = plt.subplots()
#     ax1 = sns.scatterplot(x=df2['Frequency'], y=df2['Order']*2, hue=df2['Label'], palette=_colors)

#     ax1.set_xlim(left=0, right=freq_max)
#     ax1.set_ylim(bottom=0, top=ordmax)
#     ax1.xaxis.set_major_locator(MultipleLocator(freq_max/10))
#     ax1.xaxis.set_major_formatter(FormatStrFormatter('%g'))
#     ax1.xaxis.set_minor_locator(MultipleLocator(freq_max/100))
#     ax1.set_title('''{0} - shift: {1}'''.format('Stabilization Diagram', br))
#     ax1.set_xlabel('Frequency [Hz]')
#     mplcursors.cursor()
#     plt.show()

#     Results={}
#     # if ordmin == None:
#     #     ordmin = 0
#     Results['Data'] = {'Data': data}
#     Results['Data']['Samp. Freq.'] = fs
#     Results['Data']['Ord min max'] = (0, ordmax)
#     Results['Data']['Block rows'] = br

#     Results['All Poles'] = df1
#     Results['Reduced Poles'] = df2
#     Results['Modes'] = Ms

# return fig1, Results
