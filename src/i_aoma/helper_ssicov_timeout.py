import platform
import os
import signal
import functools
import errno
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import logging
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(
    filename="iaoma_run.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode="a",
)  # Use 'w' for


class TimeoutError(Exception):
    pass


if platform.system() == "Linux":

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
else:
    import threading

    def timeout(seconds):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                result = [None]

                def target():
                    result[0] = func(*args, **kwargs)

                thread = threading.Thread(target=target)
                thread.start()
                thread.join(seconds)
                if thread.is_alive():
                    raise TimeoutError()
                return result[0]

            return wrapper

        return decorator


def run_SSICov_with_timeout(
    SingleSetup,
    qmc_limits: dict,
    timeout_seconds: int = 30,
    NumAnal: int = 0,
    set_seed=None,
    plt_stab_diag_backup: bool = False,
    output_path_stab_diag_backup: str = os.getcwd(),
    clf_model: RandomForestClassifier = None,
):
    """
    Run SSICov with timeout.
    """

    @timeout(seconds=timeout_seconds)
    def run(
        SingleSetup,
        qmc_sample,
        qmc_sample_unitary,
        NumAnal,
        plt_stab_diag_backup,
        output_path_stab_diag_backup,
    ):
        return run_SSICov(
            SingleSetup,
            qmc_sample,
            qmc_sample_unitary,
            NumAnal,
            plt_stab_diag_backup,
            output_path_stab_diag_backup,
        )

    getting_exception = 1
    IC_pred = np.nan
    discarded_qmc_samples = []
    while getting_exception:
        try:
            print(f"Run Analysis {NumAnal}....")
            qmc_sample, qmc_sample_unitary = get_qmc_sample(
                qmc_limits, set_seed=set_seed
            )

            if clf_model is not None:
                IC_pred = clf_model.predict(np.array(qmc_sample_unitary))[0]
            else:
                IC_pred = np.nan

            if (
                qmc_limits["NDOFS"] * qmc_sample[0][0]
                < qmc_sample[0][
                    1
                ]  # raise Exception("Invalid QMC sample: NDOFS*br < ordmax !")
            ) or (IC_pred == 0):
                discarded_qmc_samples.append([qmc_sample[0], qmc_sample_unitary[0]])
                print(
                    f"Analysis {NumAnal} interrupted: invalid QMC sample, e.g. NDOFS*br < ordmax or IC_pred = 0"
                )
            else:
                SingleSetup = dataslice(SingleSetup, qmc_sample)
                SSIcovresult = run(
                    SingleSetup,
                    qmc_sample,
                    qmc_sample_unitary,
                    NumAnal,
                    plt_stab_diag_backup,
                    output_path_stab_diag_backup,
                )
                getting_exception = 0
                SingleSetup = datarollback(SingleSetup)
                print(f"Analysis {NumAnal} Completed!")
                logging.info(
                    f"Analysis {NumAnal} Completed! qmc sample: {qmc_sample[0]}"
                )

                Fns_stab = np.where(
                    SSIcovresult.Lab == 1, SSIcovresult.Fn_poles, np.nan
                )
                Xi_stab = np.where(SSIcovresult.Lab == 1, SSIcovresult.Xi_poles, np.nan)
                Phi_stab = np.where(
                    SSIcovresult.Lab[:, :, np.newaxis] == 1,
                    SSIcovresult.Phi_poles,
                    np.nan,
                )
                result = {
                    # 'NumAnal': NumAnal,
                    "Fn_poles": Fns_stab,
                    # 'Lab': SSIcovresult.Lab,
                    "Xi_poles": Xi_stab,
                    "Phi_poles": Phi_stab,
                    "qmc_sample": np.array([qmc_sample[0], qmc_sample_unitary[0]]).T,
                    "IC": [IC_pred, np.nan],
                }
                return [result, discarded_qmc_samples]
        except TimeoutError:
            print(f"Analysis {NumAnal} interrupted: Timeout of the analysis!")
            discarded_qmc_samples.append([qmc_sample[0], qmc_sample_unitary[0]])
        except Exception as error:
            print(f"Analysis {NumAnal} interrupted: {str(error)}")
            discarded_qmc_samples.append([qmc_sample[0], qmc_sample_unitary[0]])


def get_qmc_sample(qmc_limits, _dim=4, _scamble=True, _numsim=1, set_seed=None):
    """
    HaltonSamples array of shape (_numsim, _dim) is generated using the Halton sequence.
    HaltonSamples[:, 0] corresponds to the timeshift parameter.
    HaltonSamples[:, 1] corresponds to the order parameter.
    HaltonSamples[:, 2] corresponds to the window_length parameter.
    HaltonSamples[:, 3] corresponds to the time_target_centering_window parameter.

    Scrambling can help distribute the points more uniformly across the space, reducing clustering and gaps.
    """

    HaltonSamples = qmc.Halton(d=_dim, scramble=_scamble, seed=set_seed).random(
        n=_numsim
    )
    par_timeshift = qmc_limits["brmin"] + np.rint(
        (qmc_limits["brmax"] - qmc_limits["brmin"]) * HaltonSamples[:, 0]
    ).astype(int)
    par_order = qmc_limits["ordmin"] + np.rint(
        (qmc_limits["ordmax"] - qmc_limits["ordmin"]) * HaltonSamples[:, 1]
    ).astype(int)
    par_window_length = qmc_limits["wlenmin"] + np.rint(
        (qmc_limits["wlenmax"] - qmc_limits["wlenmin"]) * HaltonSamples[:, 2]
    ).astype(int)
    par_time_target_centering_window = np.rint(
        HaltonSamples[:, 3] * qmc_limits["Ndata"]
    ).astype(int)

    qmc_sample_unitary = HaltonSamples.tolist()
    qmc_sample = np.vstack(
        [
            par_timeshift,
            par_order,
            par_window_length,
            par_time_target_centering_window,
        ]
    ).T
    qmc_sample = qmc_sample.tolist()

    print(
        f"QMC Sampling...Timeshift: {qmc_sample[0][0]}, Order: {qmc_sample[0][1]}, Window Length: {qmc_sample[0][2]}, Time Target Centering Window: {qmc_sample[0][3]}"
    )

    return qmc_sample, qmc_sample_unitary


def dataslice(SingleSetup, qmc_sample):
    if qmc_sample[0][3] - int(qmc_sample[0][2] / 2) < 0:
        SingleSetup.data = SingleSetup._initial_data[0 : int(qmc_sample[0][2]), :]
    elif (
        qmc_sample[0][3] + int(qmc_sample[0][2] / 2)
        > SingleSetup._initial_data.shape[0]
    ):
        SingleSetup.data = SingleSetup._initial_data[-int(qmc_sample[0][2]) :, :]
    else:
        SingleSetup.data = SingleSetup._initial_data[
            qmc_sample[0][3] - int(qmc_sample[0][2] / 2) : qmc_sample[0][3]
            + int(qmc_sample[0][2] / 2),
            :,
        ]

    SingleSetup.dt = 1 / SingleSetup.fs  # sampling interval
    SingleSetup.Nch = SingleSetup.data.shape[1]  # number of channels
    SingleSetup.Ndat = SingleSetup.data.shape[0]  # number of data points
    SingleSetup.T = SingleSetup.dt * SingleSetup.Ndat  # Period of acquisition [sec]

    SingleSetup.algorithms["SSIcov"].run_params.br = qmc_sample[0][0]
    SingleSetup.algorithms["SSIcov"].run_params.ordmax = qmc_sample[0][1]
    return SingleSetup


def datarollback(SingleSetup):
    SingleSetup.data = SingleSetup._initial_data
    SingleSetup.fs = SingleSetup._initial_fs

    SingleSetup.dt = 1 / SingleSetup.fs  # sampling interval
    SingleSetup.Nch = SingleSetup.data.shape[1]  # number of channels
    SingleSetup.Ndat = SingleSetup.data.shape[0]  # number of data points
    SingleSetup.T = SingleSetup.dt * SingleSetup.Ndat  # Period of acquisition [sec]
    return SingleSetup


def run_SSICov(
    SingleSetup,
    qmc_sample,
    qmc_sample_unitary,
    NumAnal,
    plt_stab_diag_backup: bool = False,
    output_path_phase1_stab_diag_backup: str = os.getcwd(),
):
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

    if plt_stab_diag_backup:
        fig, ax = SingleSetup.algorithms["SSIcov"].plot_stab(
            freqlim=(0, SingleSetup.fs / 2), hide_poles=False
        )
        ax.set_title(
            f"Sim. {NumAnal}, br={qmc_sample[0][0]}, ordmax={qmc_sample[0][1]}, wlen={qmc_sample_unitary[0][2]*100:.1f}$\%$, tt={qmc_sample_unitary[0][3]*100:.1f}$\%$"
        )
        plt.savefig(
            f"{output_path_phase1_stab_diag_backup+os.sep+'Stab_diag'+os.sep}{NumAnal+1000}_StabDiag.png",
            dpi=200,
        )
        plt.close()

        fig, ax = SingleSetup.algorithms["SSIcov"].plot_cluster(
            freqlim=(0, SingleSetup.fs / 2), hide_poles=False
        )
        ax.set_title(
            f"Sim. {NumAnal}, br={qmc_sample[0][0]}, ordmax={qmc_sample[0][1]}, wlen={qmc_sample_unitary[0][2]*100:.1f}$\%$, tt={qmc_sample_unitary[0][3]*100:.1f}$\%$"
        )
        plt.savefig(
            f"{output_path_phase1_stab_diag_backup+os.sep+'Damping_Freq_diag'+os.sep}{NumAnal+1000}_DampingClusters.png",
            dpi=200,
        )
        plt.close()

    return SingleSetup.algorithms["SSIcov"].result


def update_heatmap(new_x, new_y, heatmap, im, xedges, yedges):
    # global heatmap
    new_heatmap, _, _ = np.histogram2d(
        new_x, new_y, bins=[xedges, yedges]
    )  # np.histogram2d(new_x, new_y)#, \
    # bins=[round(SingleSetup.fs/2/plt_resolution['freq']),int(new_x[1,:].max()/plt_resolution['order'])], \
    # range = [[0, SingleSetup.fs/2], [new_x[1,:].min(), new_x[1,:].max()]])
    new_heatmap = new_heatmap / np.max(new_heatmap)
    heatmap += new_heatmap
    heatmap = heatmap / np.max(heatmap)  # normalize the heatmap between 0 and 1
    im.set_data(heatmap.T)
    plt.draw()
    return im, heatmap
