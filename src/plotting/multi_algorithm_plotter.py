from typing import Optional, Any, Union
from unittest import FunctionTestCase
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
import seaborn as sns
from pathlib import Path

from src.algorithms.choices import AlgorithmChoice
from src.logging.base_logger import BaseLogData
from src.core.base_optimizer import OptimizationResult
from src.logging.des_logger import DESLogData
from src.logging.mfcmaes_logger import MFCMAESLogData
from src.logging.cmaes_logger import CMAESLogData


class MultiAlgorithmPlotter:
    """Plotter for comparing multiple algorithms and their results."""

    def __init__(self, style: str = "seaborn-v0_8", figsize: tuple = (12, 8)):
        """
        Initialize the plotter.

        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        plt.style.use(style)
        sns.set_palette("husl")

    def plot_convergence_comparison(
        self,
        results: dict[AlgorithmChoice, OptimizationResult],
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Algorithm Convergence Comparison",
        log_scale: bool = True,
        show_evaluations: bool = True,
    ) -> Figure:
        """
        Plot convergence curves for multiple algorithms.

        Args:
            results: Dictionary mapping algorithm names to their results
            save_path: Path to save the plot
            title: Plot title
            log_scale: Whether to use log scale for y-axis
            show_evaluations: Whether to show x-axis as evaluations or iterations

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for algo_name, result in results.items():
            log_data = result.diagnostic

            if hasattr(log_data, "best_fitness") and log_data.best_fitness:
                x_data = (
                    log_data.evaluations
                    if show_evaluations and hasattr(log_data, "evaluations")
                    else range(len(log_data.best_fitness))
                )

                ax.plot(
                    x_data,
                    log_data.best_fitness,
                    label=f"{algo_name.value} (final: {result.best_fitness:.2e})",
                    linewidth=2,
                    alpha=0.8,
                )

        ax.set_xlabel("Function Evaluations" if show_evaluations else "Iterations")
        ax.set_ylabel("Best Fitness")
        ax.set_title(title)
        ax.legend(fontsize="x-large")
        ax.grid(True, alpha=0.3)

        if log_scale:
            ax.set_yscale("log")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_algorithm_specific_metrics(
        self,
        result: OptimizationResult,
        algorithm: AlgorithmChoice,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Plot algorithm-specific metrics.

        Args:
            result: Optimization result
            algorithm: Name of the algorithm
            save_path: Path to save the plot

        Returns:
            Figure object
        """
        log_data = result.diagnostic

        if algorithm == AlgorithmChoice.DES:
            return self._plot_des_metrics(log_data, save_path)
        if algorithm == AlgorithmChoice.MFCMAES:
            return self._plot_mfcmaes_metrics(log_data, save_path)
        if algorithm == AlgorithmChoice.CMAES:
            return self._plot_cmaes_metrics(log_data, save_path)
        return self._plot_generic_metrics(log_data, algorithm, save_path)

    def _plot_cmaes_metrics(
        self, log_data: CMAESLogData, save_path: Optional[Union[str, Path]] = None
    ) -> Figure:
        """Plot comprehensive CMA-ES-specific metrics in 4 panels."""
        fig = plt.figure(figsize=(20, 16))

        # Create 4 main panels
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

        # Use evaluations for x-axis
        evals = (
            log_data.evaluations
            if log_data.evaluations
            else range(len(log_data.best_fitness))
        )

        # ============ PANEL 1: Objective Statistics ============
        ax1 = fig.add_subplot(gs[0, 0])
        if log_data.best_fitness:
            ax1.semilogy(evals, log_data.best_fitness, "b-", linewidth=2, label="Best")
            if log_data.mean_fitness:
                ax1.semilogy(
                    evals,
                    log_data.mean_fitness,
                    "g--",
                    linewidth=1.5,
                    label="Mean f(m)",
                )
            if log_data.median_fitness:
                ax1.semilogy(
                    evals, log_data.median_fitness, "r:", linewidth=1.5, label="Median"
                )
            ax1.set_xlabel("Function Evaluations")
            ax1.set_ylabel("Fitness (log scale)")
            ax1.set_title("Convergence: Best, Mean, Median Fitness")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        if log_data.std_fitness:
            ax2.semilogy(evals, log_data.std_fitness, "purple", linewidth=2)
            ax2.set_xlabel("Function Evaluations")
            ax2.set_ylabel("Std Dev (log scale)")
            ax2.set_title("Fitness Standard Deviation")
            ax2.grid(True, alpha=0.3)

        # ============ PANEL 2: Adaptation Dynamics ============
        ax3 = fig.add_subplot(gs[1, 0])
        if log_data.sigma:
            ax3.semilogy(evals, log_data.sigma, "orange", linewidth=2)
            ax3.set_xlabel("Function Evaluations")
            ax3.set_ylabel("σ (log scale)")
            ax3.set_title("Step-Size Evolution")
            ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 1])
        if log_data.condition_number:
            ax4.semilogy(evals, log_data.condition_number, "brown", linewidth=2)
            ax4.set_xlabel("Function Evaluations")
            ax4.set_ylabel("Condition Number (log scale)")
            ax4.set_title("Covariance Condition Number (κ)")
            ax4.grid(True, alpha=0.3)

        # ============ PANEL 3: Covariance Properties ============
        ax5 = fig.add_subplot(gs[2, 0])
        if log_data.covariance_determinant:
            det_values = [d for d in log_data.covariance_determinant if d > 0]
            # Match length with evaluations
            evals_det = (
                evals[: len(det_values)] if len(det_values) < len(evals) else evals
            )
            if det_values:
                ax5.semilogy(evals_det, det_values, "teal", linewidth=2)
                ax5.set_xlabel("Function Evaluations")
                ax5.set_ylabel("det(C) (log scale)")
                ax5.set_title("Covariance Determinant (Search Volume)")
                ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[2, 1])
        if log_data.eigenvalues and len(log_data.eigenvalues) > 0:
            # Plot all eigenvalues as lines
            eigenvalues_array = np.array(log_data.eigenvalues)
            for i in range(
                min(5, eigenvalues_array.shape[1])
            ):  # Plot first 5 dimensions
                ax6.semilogy(
                    evals,
                    eigenvalues_array[:, i],
                    alpha=0.7,
                    linewidth=1.5,
                    label=f"λ_{i+1}",
                )
            ax6.set_xlabel("Function Evaluations")
            ax6.set_ylabel("Eigenvalues (log scale)")
            ax6.set_title("Eigenvalue Spectrum (first 5)")
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        # ============ PANEL 4: Evolution Paths ============
        ax7 = fig.add_subplot(gs[3, 0])
        if log_data.pc_norm and log_data.ps_norm:
            ax7.plot(evals, log_data.pc_norm, "b-", linewidth=2, label="||p_c||")
            ax7.plot(evals, log_data.ps_norm, "r--", linewidth=2, label="||p_σ||")
            ax7.set_xlabel("Function Evaluations")
            ax7.set_ylabel("Path Norm")
            ax7.set_title("Evolution Path Norms")
            ax7.legend()
            ax7.grid(True, alpha=0.3)

        ax8 = fig.add_subplot(gs[3, 1])
        if log_data.mean_vector_norm:
            ax8.semilogy(evals, log_data.mean_vector_norm, "darkgreen", linewidth=2)
            ax8.set_xlabel("Function Evaluations")
            ax8.set_ylabel("||m|| (log scale)")
            ax8.set_title("Mean Vector Norm (Distance from Origin)")
            ax8.grid(True, alpha=0.3)

        plt.suptitle("CMA-ES Comprehensive Diagnostics", fontsize=18, y=0.995)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _plot_des_metrics(
        self, log_data: DESLogData, save_path: Optional[Union[str, Path]] = None
    ) -> Figure:
        """Plot DES-specific metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # Use evaluations for x-axis
        evals = (
            log_data.evaluations
            if log_data.evaluations
            else range(len(log_data.best_fitness))
        )

        # Plot 1: Convergence
        if hasattr(log_data, "best_fitness") and log_data.best_fitness:
            axes[0].semilogy(
                evals, log_data.best_fitness, "b-", linewidth=2, label="Best Fitness"
            )
            if hasattr(log_data, "mean_fitness") and log_data.mean_fitness:
                axes[0].semilogy(
                    evals, log_data.mean_fitness, "r--", alpha=0.7, label="Mean Fitness"
                )
            axes[0].set_title("Convergence")
            axes[0].set_xlabel("Function Evaluations")
            axes[0].set_ylabel("Fitness")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        # Plot 2: Ft evolution (DES-specific)
        if hasattr(log_data, "Ft") and log_data.Ft:
            axes[1].plot(evals, log_data.Ft, "g-", linewidth=2)
            axes[1].set_title("Ft Evolution")
            axes[1].set_xlabel("Function Evaluations")
            axes[1].set_ylabel("Ft")
            axes[1].grid(True, alpha=0.3)

        # Plot 3: Condition number
        if hasattr(log_data, "condition_number") and log_data.condition_number:
            axes[2].semilogy(evals, log_data.condition_number, "m-", linewidth=2)
            axes[2].set_title("Condition Number")
            axes[2].set_xlabel("Function Evaluations")
            axes[2].set_ylabel("Condition Number")
            axes[2].grid(True, alpha=0.3)

        # Plot 4: Fitness statistics
        if (
            hasattr(log_data, "best_fitness")
            and log_data.best_fitness
            and hasattr(log_data, "worst_fitness")
            and log_data.worst_fitness
        ):
            axes[3].fill_between(
                evals,
                log_data.best_fitness,
                log_data.worst_fitness,
                alpha=0.3,
                label="Best-Worst Range",
            )
            axes[3].plot(evals, log_data.best_fitness, "b-", linewidth=2, label="Best")
            axes[3].plot(
                evals, log_data.worst_fitness, "r-", linewidth=2, label="Worst"
            )
            if hasattr(log_data, "mean_fitness") and log_data.mean_fitness:
                axes[3].plot(
                    evals, log_data.mean_fitness, "g--", linewidth=2, label="Mean"
                )
            axes[3].set_title("Fitness Statistics")
            axes[3].set_xlabel("Function Evaluations")
            axes[3].set_ylabel("Fitness")
            axes[3].set_yscale("log")
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)

        plt.suptitle("DES Algorithm Metrics", fontsize=16)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _plot_mfcmaes_metrics(
        self, log_data: MFCMAESLogData, save_path: Optional[Union[str, Path]] = None
    ) -> Figure:
        """Plot comprehensive MF-CMA-ES-specific metrics in 4 panels (similar to CMA-ES)."""
        fig = plt.figure(figsize=(20, 16))

        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

        # Use evaluations for x-axis
        evals = (
            log_data.evaluations
            if log_data.evaluations
            else range(len(log_data.best_fitness))
        )

        # ============ PANEL 1: Objective Statistics ============
        ax1 = fig.add_subplot(gs[0, 0])
        if log_data.best_fitness:
            ax1.semilogy(evals, log_data.best_fitness, "b-", linewidth=2, label="Best")
            if log_data.midpoint_fitness:
                ax1.semilogy(
                    evals,
                    log_data.midpoint_fitness,
                    "g--",
                    linewidth=1.5,
                    label="Midpoint f(m)",
                )
            if log_data.mean_fitness:
                ax1.semilogy(
                    evals,
                    log_data.mean_fitness,
                    "r:",
                    linewidth=1.5,
                    label="Mean",
                )
            ax1.set_xlabel("Function Evaluations")
            ax1.set_ylabel("Fitness (log scale)")
            ax1.set_title("Convergence: Best, Midpoint, Mean Fitness")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        if log_data.std_fitness:
            ax2.semilogy(evals, log_data.std_fitness, "purple", linewidth=2)
            ax2.set_xlabel("Function Evaluations")
            ax2.set_ylabel("Std Dev (log scale)")
            ax2.set_title("Fitness Standard Deviation")
            ax2.grid(True, alpha=0.3)

        # ============ PANEL 2: Step-Size Adaptation (PPMF) ============
        ax3 = fig.add_subplot(gs[1, 0])
        if log_data.sigma:
            ax3.semilogy(evals, log_data.sigma, "orange", linewidth=2)
            ax3.set_xlabel("Function Evaluations")
            ax3.set_ylabel("σ (log scale)")
            ax3.set_title("Step-Size Evolution (PPMF)")
            ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 1])
        if log_data.p_succ:
            ax4.plot(evals, log_data.p_succ, "m-", linewidth=2, label="p_succ")
            ax4.axhline(
                0.2,
                color="k",
                linestyle="--",
                linewidth=1,
                label=f"Target ({0.2})",
            )
            ax4.set_xlabel("Function Evaluations")
            ax4.set_ylabel("Success Probability")
            ax4.set_title("PPMF Success Probability")
            ax4.set_ylim([0, 1])
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # ============ PANEL 3: Archive and Constraint Info ============
        ax5 = fig.add_subplot(gs[2, 0])
        if log_data.constraint_violations:
            ax5.plot(evals, log_data.constraint_violations, "red", linewidth=2)
            ax5.set_xlabel("Function Evaluations")
            ax5.set_ylabel("Violations")
            ax5.set_title("Constraint Violations")
            ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[2, 1])
        if log_data.best_fitness and log_data.worst_fitness:
            ax6.semilogy(evals, log_data.best_fitness, "b-", linewidth=2, label="Best")
            ax6.semilogy(
                evals, log_data.worst_fitness, "r-", linewidth=2, label="Worst"
            )
            if log_data.mean_fitness:
                ax6.semilogy(
                    evals, log_data.mean_fitness, "g--", linewidth=1.5, label="Mean"
                )
            ax6.set_xlabel("Function Evaluations")
            ax6.set_ylabel("Fitness (log scale)")
            ax6.set_title("Fitness Statistics (Best/Mean/Worst)")
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        # ============ PANEL 4: Evolution Path and Mean Vector ============
        ax7 = fig.add_subplot(gs[3, 0])
        if log_data.pc_norm:
            ax7.plot(evals, log_data.pc_norm, "b-", linewidth=2, label="||p_c||")
            ax7.set_xlabel("Function Evaluations")
            ax7.set_ylabel("Path Norm")
            ax7.set_title("Evolution Path Norm")
            ax7.legend()
            ax7.grid(True, alpha=0.3)

        ax8 = fig.add_subplot(gs[3, 1])
        if log_data.mean_vector_norm:
            ax8.semilogy(evals, log_data.mean_vector_norm, "darkgreen", linewidth=2)
            ax8.set_xlabel("Function Evaluations")
            ax8.set_ylabel("||m|| (log scale)")
            ax8.set_title("Mean Vector Norm (Distance from Origin)")
            ax8.grid(True, alpha=0.3)

        plt.suptitle("MF-CMA-ES Comprehensive Diagnostics", fontsize=18, y=0.995)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _plot_generic_metrics(
        self,
        log_data: BaseLogData,
        algorithm: AlgorithmChoice,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Plot generic metrics for unknown algorithms."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # Use evaluations for x-axis
        evals = (
            log_data.evaluations
            if hasattr(log_data, "evaluations") and log_data.evaluations
            else range(len(log_data.best_fitness))
        )

        # Plot 1: Convergence
        if hasattr(log_data, "best_fitness") and log_data.best_fitness:
            axes[0].semilogy(evals, log_data.best_fitness, "b-", linewidth=2)
            axes[0].set_title("Best Fitness Evolution")
            axes[0].set_xlabel("Function Evaluations")
            axes[0].set_ylabel("Best Fitness")
            axes[0].grid(True, alpha=0.3)

        # Plot 2: Fitness statistics
        if hasattr(log_data, "best_fitness") and log_data.best_fitness:
            if hasattr(log_data, "mean_fitness") and log_data.mean_fitness:
                axes[1].plot(
                    evals, log_data.mean_fitness, "g-", linewidth=2, label="Mean"
                )
            if hasattr(log_data, "std_fitness") and log_data.std_fitness:
                axes[1].plot(
                    evals, log_data.std_fitness, "r--", linewidth=2, label="Std Dev"
                )
            axes[1].set_title("Fitness Statistics")
            axes[1].set_xlabel("Function Evaluations")
            axes[1].set_ylabel("Fitness")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        # Plot 3: Condition number (if available)
        if hasattr(log_data, "condition_number") and log_data.condition_number:
            axes[2].semilogy(evals, log_data.condition_number, "m-", linewidth=2)
            axes[2].set_title("Condition Number")
            axes[2].set_xlabel("Function Evaluations")
            axes[2].set_ylabel("Condition Number")
            axes[2].grid(True, alpha=0.3)

        # Plot 4: Available custom metric (algorithm-specific)
        custom_metric = self._find_custom_metric(log_data)
        if custom_metric:
            metric_name, metric_data = custom_metric
            axes[3].plot(evals, metric_data, "c-", linewidth=2)
            axes[3].set_title(f'{metric_name.replace("_", " ").title()}')
            axes[3].set_xlabel("Function Evaluations")
            axes[3].set_ylabel(metric_name.replace("_", " ").title())
            axes[3].grid(True, alpha=0.3)

        plt.suptitle(f"{algorithm.value} Algorithm Metrics", fontsize=16)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _find_custom_metric(
        self, log_data: BaseLogData
    ) -> Optional[tuple[str, list[Any]]]:
        """Find the first available custom metric in log data."""
        common_attrs = {
            "iteration",
            "evaluations",
            "best_fitness",
            "worst_fitness",
            "mean_fitness",
            "std_fitness",
            "population",
            "best_solution",
            "eigenvalues",
            "condition_number",
        }

        for attr_name in dir(log_data):
            if (
                not attr_name.startswith("_")
                and attr_name not in common_attrs
                and hasattr(log_data, attr_name)
            ):

                attr_value = getattr(log_data, attr_name)
                if isinstance(attr_value, list) and len(attr_value) > 0:
                    return attr_name, attr_value

        return None

    def plot_parameter_evolution(
        self,
        results: dict[AlgorithmChoice, OptimizationResult],
        parameter_name: str,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None,
    ) -> Figure:
        """
        Plot evolution of a specific parameter across algorithms.

        Args:
            results: Dictionary of algorithm results
            parameter_name: Name of parameter to plot
            save_path: Path to save the plot
            title: Custom plot title

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for algo_name, result in results.items():
            log_data = result.diagnostic

            if hasattr(log_data, parameter_name):
                param_data = getattr(log_data, parameter_name)
                if isinstance(param_data, list) and len(param_data) > 0:
                    # Use evaluations for x-axis
                    evals = (
                        log_data.evaluations
                        if hasattr(log_data, "evaluations") and log_data.evaluations
                        else range(len(param_data))
                    )
                    ax.plot(
                        evals, param_data, label=algo_name.value, linewidth=2, alpha=0.8
                    )

        ax.set_xlabel("Function Evaluations")
        ax.set_ylabel(parameter_name.replace("_", " ").title())
        ax.set_title(title or f'{parameter_name.replace("_", " ").title()} Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_summary_report(
        self,
        results: dict[str, OptimizationResult],
        save_dir: Optional[Union[str, Path]] = None,
    ) -> dict[str, Figure]:
        """
        Create a comprehensive summary report with multiple plots.

        Args:
            results: Dictionary of algorithm results
            save_dir: Directory to save plots

        Returns:
            Dictionary of figure objects
        """
        figures = {}

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

        # Convergence comparison
        conv_path = save_dir / "convergence_comparison.png" if save_dir else None
        figures["convergence"] = self.plot_convergence_comparison(results, conv_path)

        for algo_name, result in results.items():
            algorithm = AlgorithmChoice(algo_name)
            algo_path = (
                save_dir / f"{algo_name.lower()}_metrics.png" if save_dir else None
            )
            figures[f"{algo_name}_metrics"] = self.plot_algorithm_specific_metrics(
                result, algorithm, algo_path
            )

        return figures
