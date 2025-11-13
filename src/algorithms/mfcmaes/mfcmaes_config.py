from dataclasses import dataclass, field
import numpy as np

from core.config_base import BaseConfig


def default_mfcmaes_population_size(obj: "MFCMAESConfig") -> int:
    """Default population size for MFCMAES based on dimensions."""
    return 4 + int(3 * np.log(obj.dimensions))


def default_mfcmaes_budget(obj: "MFCMAESConfig") -> int:
    """Default budget for MFCMAES based on dimensions."""
    return 1000 * obj.dimensions


@dataclass
class MFCMAESConfig(BaseConfig):
    # Step-size (sigma) and PPMF parameters
    sigma: float = 1.0
    p_target_ppmf: float = 0.2
    damps_ppmf: float = 2.0

    # Stopping conditions
    stop_fitness: float = 1e-8
    max_iterations: int = field(init=False)

    # Internal parameters (CMA-ES defaults)
    mu: int | None = None  # if None, floor(lambda_/2)
    equal_weights: bool = True  # equal recombination weights (matches R code)
    trace: bool = False
    keep_best: bool = True
    flatland_escape: bool = True

    # Archive window length h (recommended: floor(1.4*n) + 20)
    window: int | None = None

    # Random seed (optional)
    seed: int | None = None

    def __post_init__(self):
        # Defaults based on dimensions
        self.population_size = default_mfcmaes_population_size(self)
        self.budget = default_mfcmaes_budget(self)
        self.max_iterations = int(np.round(self.budget / max(1, self.population_size)))

        # Window default
        if self.window is None:
            self.window = int(np.floor(1.4 * self.dimensions) + 20)

        # Diagnostics: BaseConfig already has diag_* switches
        # You can enable self.diag_enabled and related flags via config.enable_all_diagnostics()

        # Validation
        self.validate()
