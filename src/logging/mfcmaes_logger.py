from typing import Any
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field

from src.algorithms.choices import AlgorithmChoice
from src.logging.base_logger import BaseLogger, BaseLogData
from src.algorithms.mfcmaes.mfcmaes_config import MFCMAESConfig


@dataclass
class MFCMAESLogData(BaseLogData):
    """MF-CMA-ES-specific log data."""

    sigma: list[float] = field(default_factory=list)
    p_succ: list[float] = field(default_factory=list)
    midpoint_fitness: list[float] = field(default_factory=list)
    constraint_violations: list[int] = field(default_factory=list)

    def clear(self) -> None:
        self.clear_common()
        self.sigma.clear()
        self.p_succ.clear()
        self.midpoint_fitness.clear()
        self.constraint_violations.clear()

    def to_dict(self) -> dict[str, list[Any]]:
        result = self.to_dict_common()
        result.update(
            {
                "sigma": self.sigma,
                "p_succ": self.p_succ,
                "midpoint_fitness": self.midpoint_fitness,
                "constraint_violations": self.constraint_violations,
            }
        )
        return result


class MFCMAESLogger(BaseLogger[MFCMAESLogData]):
    """Logger for MF-CMA-ES algorithm."""

    def __init__(self, config: MFCMAESConfig):
        super().__init__(config, AlgorithmChoice.MFCMAES)

    def _create_log_data(self) -> MFCMAESLogData:
        return MFCMAESLogData()

    def log_iteration(
        self,
        iteration: int,
        evaluations: int,
        best_fitness: float,
        worst_fitness: float,
        mean_fitness: float,
        sigma: float,
        p_succ: float,
        midpoint_fitness: float,
        constraint_violations: int,
        fitness: NDArray[np.float64] | None = None,
        population: NDArray[np.float64] | None = None,
        best_solution: NDArray[np.float64] | None = None,
        eigenvalues: NDArray[np.float64] | None = None,
        **kwargs,
    ) -> None:
        self.logs.iteration.append(iteration)
        self.logs.evaluations.append(evaluations)
        self.logs.best_fitness.append(best_fitness)
        self.logs.worst_fitness.append(worst_fitness)
        self.logs.mean_fitness.append(mean_fitness)
        self.logs.sigma.append(sigma)
        self.logs.p_succ.append(p_succ)
        self.logs.midpoint_fitness.append(midpoint_fitness)
        self.logs.constraint_violations.append(constraint_violations)

        if fitness is not None:
            self.logs.std_fitness.append(float(np.std(fitness)))
        else:
            self.logs.std_fitness.append(0.0)

        if population is not None and self.config.diag_pop:
            self.logs.population.append(population.copy())

        if best_solution is not None:
            self.logs.best_solution.append(best_solution.copy())

        if eigenvalues is not None and self.config.diag_eigen:
            self.logs.eigenvalues.append(eigenvalues.copy())
            if len(eigenvalues) > 0:
                self.logs.condition_number.append(
                    eigenvalues[0] / max(1e-300, eigenvalues[-1])
                )
