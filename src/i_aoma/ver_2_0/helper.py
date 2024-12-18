import os
import signal
import functools
import errno

# import time
# import numpy as np
# from multiprocessing import shared_memory
import matplotlib
import matplotlib.pyplot as plt

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
        f"Sim. {NumAnal}, br={qmc_sample[0][0]}, ordmax={qmc_sample[0][1]}, wlen={qmc_sample_unitary[0][2]*100:.1f}$\%$, tt={qmc_sample_unitary[0][3]*100:1f}$\%$"
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
        f"Sim. {NumAnal}, br={qmc_sample[0][0]}, ordmax={qmc_sample[0][1]}, wlen={qmc_sample_unitary[0][2]*100:.1f}$\%$, tt={qmc_sample_unitary[0][3]*100:1f}$\%$"
    )
    plt.savefig(
        f"src/i_aoma/ver_2_0/Results/trave1_results_1cuscino/{NumAnal+1000}_DampingClusters.png",
        dpi=300,
    )
    plt.close()

    return SingleSetup.algorithms["SSIcov"].result
