import matplotlib.pyplot as plt
from pathlib import Path

from src import AlgorithmFactory

from src.algorithms.choices import AlgorithmChoice
from src.algorithms.des.config import DESConfig
from src.utils.boundary_handlers import BoundaryHandlerType
from src.utils.benchmark_functions import CEC17Function, Sphere
from src.utils.initial_point_generator import (
    InitialPointGenerator,
    InitialPointGeneratorType,
)
from src.plotting.multi_algorithm_plotter import MultiAlgorithmPlotter

import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="opfunu")

plt.ioff()
plt.switch_backend("Agg")


def run_optimization_example():
    """Run a simple optimization example using the new architecture."""

    dimensions = 10

    # opt_func = CEC17Function(dimensions=dimensions, function_id=3)
    opt_func = Sphere(dimensions=dimensions)

    lower_bounds = -100.12
    upper_bounds = 100.12

    initial_point_generator = InitialPointGenerator(
        strategy=InitialPointGeneratorType.UNIFORM,
        dimensions=dimensions,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    initial_point = initial_point_generator.generate()

    config = DESConfig(dimensions=dimensions)

    config.budget = 10000 * dimensions
    config.population_size = 4 * dimensions

    config.enable_all_diagnostics()

    print("Starting DES optimization...")
    print(f"Dimensions: {dimensions}")
    print(f"Budget: {config.budget}")
    print(f"Population size: {config.population_size}")
    print(f"Initial point value: {opt_func(initial_point):.20f}")
    print(f"Configuration: {config}")

    optimizer = AlgorithmFactory.create_optimizer(
        algorithm=AlgorithmChoice.DES,
        func=opt_func,
        initial_point=initial_point,
        config=config,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        boundary_strategy=BoundaryHandlerType.CLAMP,
    )

    result = optimizer.optimize()

    print("\nOptimization completed:")
    print(f"Best fitness: {result.best_fitness:.20f}")
    print(f"Function evaluations: {result.evaluations}")
    print(f"Message: {result.message}")
    print(f"Algorithm: {result.algorithm}")

    if hasattr(result.diagnostic, "Ft"):
        des_logs = result.diagnostic
        if des_logs.Ft:
            print(f"Final Ft value: {des_logs.Ft[-1]}")

    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    print(f"\nSaving plots to: {output_dir.absolute()}")

    plotter = MultiAlgorithmPlotter()

    metrics_path = output_dir / "des_metrics.png"
    _ = plotter.plot_algorithm_specific_metrics(
        result, AlgorithmChoice.DES, save_path=metrics_path
    )

    print(f"Saved DES metrics plot to: {metrics_path}")

    results_dict = {AlgorithmChoice.DES: result}
    convergence_path = output_dir / "convergence_comparison.png"
    _ = plotter.plot_convergence_comparison(
        results_dict,
        save_path=convergence_path,
        title="DES Convergence on Sphere Function",
    )
    print(f"Saved convergence plot to: {convergence_path}")

    ft_path = output_dir / "ft_evolution.png"
    _ = plotter.plot_parameter_evolution(
        results_dict, "Ft", save_path=ft_path, title="DES Ft Parameter Evolution"
    )

    print(f"\nAll plots saved successfully to: {output_dir.absolute()}")


if __name__ == "__main__":
    run_optimization_example()
