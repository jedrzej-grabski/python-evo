import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Any

from src.algorithms.choices import AlgorithmChoice
from src.logging.base_logger import BaseLogger, BaseLogData
from src.algorithms.cmaes.cmaes_config import CMAESConfig


@dataclass
class CMAESLogData(BaseLogData):
    sigma: list[float] = field(default_factory=list)
    eigenvalues: list[NDArray[np.float64]] = field(default_factory=list)
    bestVal_history: list[float] = field(default_factory=list)

    def clear(self) -> None:
        self.clear_common()
        self.sigma.clear()
        self.eigenvalues.clear()
        self.bestVal_history.clear()

    def to_dict(self) -> dict[str, list[Any]]:
        d = self.to_dict_common()
        d.update(
            {
                "sigma": self.sigma,
                "eigenvalues": self.eigenvalues,
                "bestVal_history": self.bestVal_history,
            }
        )
        return d


class CMAESLogger(BaseLogger[CMAESLogData]):
    def __init__(self, config: CMAESConfig):
        super().__init__(config, AlgorithmChoice.CMAES)

    def _create_log_data(self) -> CMAESLogData:
        return CMAESLogData()

    def log_iteration(
        self,
        iteration: int,
        evaluations: int,
        best_fitness: float,
        worst_fitness: float,
        mean_fitness: float,
        sigma: float,
        fitness: NDArray[np.float64] | None = None,
        population: NDArray[np.float64] | None = None,
        best_solution: NDArray[np.float64] | None = None,
        eigenvalues: NDArray[np.float64] | None = None,
    ) -> None:
        self.logs.iteration.append(iteration)
        self.logs.evaluations.append(evaluations)
        self.logs.best_fitness.append(best_fitness)
        self.logs.worst_fitness.append(worst_fitness)
        self.logs.mean_fitness.append(mean_fitness)
        if fitness is not None:
            self.logs.std_fitness.append(float(np.std(fitness)))
        else:
            self.logs.std_fitness.append(0.0)
        self.logs.sigma.append(sigma)
        if eigenvalues is not None and self.config.diag_eigen:
            self.logs.eigenvalues.append(eigenvalues.copy())
            if len(eigenvalues) > 1:
                self.logs.condition_number.append(
                    eigenvalues[0] / max(1e-300, eigenvalues[-1])
                )
        if population is not None and self.config.diag_pop:
            self.logs.population.append(population.copy())
        if best_solution is not None:
            self.logs.best_solution.append(best_solution.copy())
        self.logs.bestVal_history.append(best_fitness)
