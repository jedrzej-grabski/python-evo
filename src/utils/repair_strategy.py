from enum import Enum
from typing import Tuple

from numpy.typing import NDArray
import numpy as np

from src.utils.boundary_handlers import BoundaryHandler
from src.utils.helpers import delete_inf_nan


class RepairStrategyType(Enum):
    LAMARCKIAN = "lamarckian"
    NON_LAMARCKIAN = "non_lamarckian"


class RepairStrategy:
    """
    Encapsulates the repair strategy (Lamarckian vs non-Lamarckian)
    for handling boundary violations in evolutionary algorithms.
    """

    def __init__(
        self, strategy_type: RepairStrategyType, boundary_handler: BoundaryHandler
    ) -> None:
        self.strategy_type = strategy_type
        self.boundary_handler = boundary_handler

    def is_lamarckian(self) -> bool:
        return self.strategy_type == RepairStrategyType.LAMARCKIAN

    def repair_population(
        self, population: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
        """
        Repair the population according to the selected strategy.

        Args:
            population: The population to repair (shape: lambda_, N)

        Returns:
            Tuple of (effective_population, repaired_population, repair_count)
            - effective_population: Population used for future generations
            - repaired_population: Population used for fitness evaluation
            - repair_count: Number of individuals that needed repair
        """
        population_temp = population.copy()
        population_repaired = np.array(
            [self.boundary_handler.repair(individual) for individual in population]
        )

        # Count repaired individuals
        counter_repaired = 0
        for i in range(population.shape[0]):
            if not np.array_equal(population_temp[i], population_repaired[i]):
                counter_repaired += 1

        # In Lamarckian approach, we replace the original population with repaired one
        effective_population = (
            population_repaired.copy() if self.is_lamarckian() else population.copy()
        )

        return effective_population, population_repaired, counter_repaired

    def apply_fitness_strategy(
        self,
        population: NDArray[np.float64],
        repaired_population: NDArray[np.float64],
        fitness: NDArray[np.float64],
        worst_fitness: float,
    ) -> NDArray[np.float64]:
        """
        Apply the fitness strategy based on repair approach.

        Args:
            population: Original population
            repaired_population: Repaired population
            fitness: Fitness values from evaluating repaired population
            worst_fitness: Worst fitness seen so far

        Returns:
            Adjusted fitness values based on strategy
        """
        if self.is_lamarckian():
            return fitness
        else:
            return self._apply_penalty(
                population, repaired_population, fitness, worst_fitness
            )

    def _apply_penalty(
        self,
        population: NDArray[np.float64],
        repaired_population: NDArray[np.float64],
        fitness: NDArray[np.float64],
        worst_fitness: float,
    ) -> NDArray[np.float64]:
        """Apply penalty to solutions that violated constraints."""
        # Find individuals that needed repair
        needs_repair = np.zeros(population.shape[0], dtype=bool)
        for i in range(population.shape[0]):
            needs_repair[i] = not np.array_equal(population[i], repaired_population[i])

        # Calculate squared distances between original and repaired
        sq_distances = np.zeros(population.shape[0])
        for i in range(population.shape[0]):
            if needs_repair[i]:
                sq_distances[i] = np.sum((population[i] - repaired_population[i]) ** 2)

        # Apply penalty based on distance
        penalized_fitness = fitness.copy()
        penalized_fitness[needs_repair] = worst_fitness + sq_distances[needs_repair]

        return delete_inf_nan(penalized_fitness)

    def get_best_solution(
        self,
        population: NDArray[np.float64],
        repaired_population: NDArray[np.float64],
        best_idx: int,
    ) -> NDArray[np.float64]:
        """Get the best solution based on strategy."""
        return (
            population[best_idx]
            if self.is_lamarckian()
            else repaired_population[best_idx]
        )
