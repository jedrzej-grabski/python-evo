from dataclasses import dataclass, field
import numpy as np

from src.core.config_base import BaseConfig


@dataclass
class CMAESConfig(BaseConfig):
    sigma: float = 1.0
    stop_fitness: float = 1e-8
    max_iterations: int = field(init=False)

    # Strategy parameters
    lambda_: int | None = None
    mu: int | None = None
    weights: np.ndarray | None = None
    equal_weights: bool = True

    # Step-size parameters (CSA)
    cs: float | None = None
    damps: float | None = None
    cc: float | None = None
    ccov_mu: float | None = None
    ccov_1: float | None = None

    # Termination toggles
    terminate_stopfitness: bool = True
    terminate_std_dev_tol: bool = True
    terminate_cov_mat_cond: bool = True
    terminate_maxiter: bool = True
    tolx_factor: float = 1e-12

    # Hacks
    flatland_escape: bool = True
    do_hsig: bool = True
    trace: bool = False
    keep_best: bool = True

    # Diagnostics
    diag_sigma: bool = False
    diag_eigen: bool = False
    diag_value: bool = False
    diag_pop: bool = False
    diag_bestVal: bool = True

    def __post_init__(self):
        N = self.dimensions
        if self.lambda_ is None:
            self.lambda_ = 4 * N
        if self.mu is None:
            self.mu = int(np.floor(self.lambda_ / 2))
        if self.weights is None:
            if self.equal_weights:
                self.weights = np.ones(self.mu) / self.mu
            else:
                raw = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
                self.weights = raw / np.sum(raw)
        mueff = (np.sum(self.weights) ** 2) / np.sum(self.weights**2)

        if self.cc is None:
            self.cc = 4.0 / (N + 4.0)
        if self.cs is None:
            self.cs = (mueff + 2.0) / (N + mueff + 3.0)
        if self.damps is None:
            self.damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (N + 1)) - 1) + self.cs
        if self.ccov_mu is None:
            self.ccov_mu = mueff
        if self.ccov_1 is None:
            self.ccov_1 = (1 / self.ccov_mu) * 2 / (N + 1.4) ** 2 + (
                1 - 1 / self.ccov_mu
            ) * ((2 * self.ccov_mu - 1) / ((N + 2) ** 2 + 2 * self.ccov_mu))

        self.population_size = self.lambda_
        self.budget = 10000 * N
        self.max_iterations = int(round(self.budget / self.lambda_))
