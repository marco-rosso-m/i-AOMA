from .IAOMAPhase1 import IAOMAPhase1


class IAOMAPhase2(IAOMAPhase1):
    """
    Child class for Phase 2 operations.
    Inherits attributes from IAOMA and methods from IAOMAPhase1.
    Implements new functionality specific to Phase 2.
    """

    def __init__(self):
        # Inherit attributes from IAOMAPhase1 (and IAOMA indirectly)
        super().__init__()

    def loop_phase2_operations(self):
        """
        Loop operations until convergence.
        """
        print("Running Phase 2 operations with convergence checks...")
        self.qmc_sample()  # Reuse method from IAOMAPhase1
        self.detrend_and_decimate()
        self.SSICov()
        self.check_convergence()
        print("Phase 2 operations completed.")

    def check_convergence(self):
        print("Checking convergence of results...")
        super().plot_ic_graph()
