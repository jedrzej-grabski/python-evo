from dataclasses import dataclass, field
from typing import Any, Protocol
import numpy as np
from numpy.typing import NDArray

from src.algorithms.choices import AlgorithmChoice


class ConfigProtocol(Protocol):
    """Protocol defining the interface for algorithm configurations."""

    dimensions: int
    budget: int
    population_size: int

    def validate(self) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...


@dataclass
class BaseConfig:
    """
    Base configuration class for optimization algorithms.
    Contains common parameters shared across different algorithms.
    """

    dimensions: int
    """Number of dimensions in the problem"""

    budget: int = field(default=0)
    """Maximum number of function evaluations"""

    population_size: int = field(default=0)
    """Size of the population"""

    # Diagnostic logging options (common across algorithms)
    diag_enabled: bool = False
    """Enable all diagnostics"""

    diag_value: bool = False
    """Log population fitness values"""

    diag_mean: bool = False
    """Log mean fitness"""

    diag_meanCords: bool = False
    """Log mean coordinates"""

    diag_pop: bool = False
    """Log populations"""

    diag_bestVal: bool = True
    """Log best fitness"""

    diag_worstVal: bool = False
    """Log worst fitness"""

    diag_eigen: bool = False
    """Log eigenvalues"""

    def validate(self) -> None:
        """Validate the configuration parameters."""
        if self.dimensions <= 0:
            raise ValueError("Dimensions must be a positive integer.")
        if self.budget <= 0:
            raise ValueError("Budget must be a positive integer.")
        if self.population_size <= 0:
            raise ValueError("Population size must be a positive integer.")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration parameters to a dictionary."""
        return {
            "dimensions": self.dimensions,
            "budget": self.budget,
            "population_size": self.population_size,
            "diag_enabled": self.diag_enabled,
            "diag_value": self.diag_value,
            "diag_mean": self.diag_mean,
            "diag_meanCords": self.diag_meanCords,
            "diag_pop": self.diag_pop,
            "diag_bestVal": self.diag_bestVal,
            "diag_worstVal": self.diag_worstVal,
            "diag_eigen": self.diag_eigen,
        }

    def enable_all_diagnostics(self) -> None:
        """Enable all diagnostic logging options."""
        self.diag_enabled = True
        self.diag_value = True
        self.diag_mean = True
        self.diag_meanCords = True
        self.diag_pop = True
        self.diag_bestVal = True
        self.diag_worstVal = True
        self.diag_eigen = True

    def disable_all_diagnostics(self) -> None:
        """Disable all diagnostic logging options."""
        self.diag_enabled = False
        self.diag_value = False
        self.diag_mean = False
        self.diag_meanCords = False
        self.diag_pop = False
        self.diag_bestVal = False
        self.diag_worstVal = False
        self.diag_eigen = False

    def with_convergence_diagnostics(self) -> None:
        """Enable only diagnostics needed for convergence plots."""
        self.disable_all_diagnostics()
        self.diag_bestVal = True
