import numpy as np
from scipy.stats import qmc
from joblib import Parallel, delayed

from .helper import timeout, run_SSICov

from memory_profiler import profile


class IAOMAPhase1:
    """
    Child class for Phase 1 operations.
    Inherits attributes from IAOMA but not its methods.
    """

    def __init__(self, iaoma):
        # Inherit attributes from IAOMA
        self.SingleSetup = iaoma.SingleSetup
        self.Ndata = iaoma.Ndata
        self.brmin = iaoma.brmin
        self.brmax = iaoma.brmax
        self.wlenmin = iaoma.wlenmin
        self.wlenmax = iaoma.wlenmax
        self.ordmin = iaoma.ordmin
        self.ordmax = iaoma.ordmax

    def loop_phase1_operations(
        self, NsimPh1: int, n_jobs: int = -1, timeout_seconds: int = 30, set_seed=None
    ):
        @profile(
            precision=4,
            stream=open(
                f"src/i_aoma/ver_2_0/memprof_an_{NsimPh1}_njobs_{n_jobs}.log", "a+"
            ),
        )
        def run():
            """
            Loop operations until collecting NsimPh1 results.
            """
            self.set_seed = set_seed
            self.n_jobs = n_jobs

            # Sequential loop
            if n_jobs == 0:
                print("Running IAOMA-Phase 1 (sequential mode)...")
                # Nsim_results = []
                # for NumAnal in range( NsimPh1 ):
                # getting_exception = 1
                # error_qmc_samples = []
                # while getting_exception :
                #     try:
                #         print(f"Run Analysis {NumAnal}....")
                #         # qmc_sample, qmc_sample_unitary = self.qmc_sample(set_seed=self.set_seed)
                #         result = self.run_SSICov_with_timeout(timeout_seconds, NumAnal)
                #         getting_exception = 0
                #         print(f"Analysis {NumAnal} Completed!")
                #         # Nsim_results.append([result, qmc_sample, error_qmc_samples])
                #     except TimeoutError:
                #         print(f"Analysis {NumAnal} interrupted: Timeout of the analysis!")
                #         error_qmc_samples.append(qmc_sample)
                #     except Exception as error:
                #         print(f"Analysis {NumAnal} interrupted: {str(error)}")
                #         error_qmc_samples.append(qmc_sample)

            else:
                # pass
                print("Running IAOMA-Phase 1 (parallel mode)...")
                results = Parallel(n_jobs=n_jobs)(
                    delayed(run_SSICov_with_timeout)(self, timeout_seconds, NumAnal)
                    for NumAnal in range(NsimPh1)
                )

            print("Phase 1 operations completed.")
            return results

        return run()

    def qmc_sample(self, _dim=4, _scamble=True, _numsim=1, set_seed=None):
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
        par_timeshift = self.brmin + np.rint(
            (self.brmax - self.brmin) * HaltonSamples[:, 0]
        ).astype(int)
        par_order = self.ordmin + np.rint(
            (self.ordmax - self.ordmin) * HaltonSamples[:, 1]
        ).astype(int)
        par_window_length = self.wlenmin + np.rint(
            (self.wlenmax - self.wlenmin) * HaltonSamples[:, 2]
        ).astype(int)
        par_time_target_centering_window = np.rint(
            HaltonSamples[:, 3] * self.Ndata
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

    def dataslice(self, qmc_sample):
        if qmc_sample[0][3] - int(qmc_sample[0][2] / 2) < 0:
            self.SingleSetup.data = self.SingleSetup._initial_data[
                0 : int(qmc_sample[0][2]), :
            ]
        elif (
            qmc_sample[0][3] + int(qmc_sample[0][2] / 2)
            > self.SingleSetup._initial_data.shape[0]
        ):
            self.SingleSetup.data = self.SingleSetup._initial_data[
                -int(qmc_sample[0][2]) :, :
            ]
        else:
            self.SingleSetup.data = self.SingleSetup._initial_data[
                qmc_sample[0][3] - int(qmc_sample[0][2] / 2) : qmc_sample[0][3]
                + int(qmc_sample[0][2] / 2),
                :,
            ]

        self.SingleSetup.dt = 1 / self.SingleSetup.fs  # sampling interval
        self.SingleSetup.Nch = self.SingleSetup.data.shape[1]  # number of channels
        self.SingleSetup.Ndat = self.SingleSetup.data.shape[0]  # number of data points
        self.SingleSetup.T = (
            self.SingleSetup.dt * self.SingleSetup.Ndat
        )  # Period of acquisition [sec]

        self.SingleSetup.algorithms["SSIcov"].run_params.br = qmc_sample[0][0]
        self.SingleSetup.algorithms["SSIcov"].run_params.ordmax = qmc_sample[0][1]

    def datarollback(self):
        self.SingleSetup.data = self.SingleSetup._initial_data
        self.SingleSetup.fs = self.SingleSetup._initial_fs

        self.SingleSetup.dt = 1 / self.SingleSetup.fs  # sampling interval
        self.SingleSetup.Nch = self.SingleSetup.data.shape[1]  # number of channels
        self.SingleSetup.Ndat = self.SingleSetup.data.shape[0]  # number of data points
        self.SingleSetup.T = (
            self.SingleSetup.dt * self.SingleSetup.Ndat
        )  # Period of acquisition [sec]


# @profile
def run_SSICov_with_timeout(self, timeout_seconds, NumAnal):
    # @profile(precision=4, stream= open(f"src/i_aoma/ver_2_0/memprof_an_{NumAnal}_njobs_{self.n_jobs}.log", "a+"))
    @timeout(seconds=timeout_seconds)  # Default timeout of 30 seconds
    def run(SingleSetup, qmc_sample, qmc_sample_unitary, NumAnal):
        return run_SSICov(SingleSetup, qmc_sample, qmc_sample_unitary, NumAnal)

    # try:
    #     print(f"Run Analysis {NumAnal}....")
    #     print(qmc_sample)
    #     return run()
    # except Exception as e:
    #     return print(f"Analysis {NumAnal} interrupted: Timeout of the analysis!")

    getting_exception = 1
    error_qmc_samples = []
    while getting_exception:
        try:
            print(f"Run Analysis {NumAnal}....")
            qmc_sample, qmc_sample_unitary = self.qmc_sample(set_seed=self.set_seed)

            self.dataslice(qmc_sample)

            result = run(self.SingleSetup, qmc_sample, qmc_sample_unitary, NumAnal)

            getting_exception = 0

            self.datarollback()

            print(f"Analysis {NumAnal} Completed!")
            return [result, qmc_sample[0], qmc_sample_unitary[0], error_qmc_samples]
        except TimeoutError:
            print(f"Analysis {NumAnal} interrupted: Timeout of the analysis!")
            error_qmc_samples.append(qmc_sample[0])
        except Exception as error:
            print(f"Analysis {NumAnal} interrupted: {str(error)}")
            error_qmc_samples.append(qmc_sample[0])
