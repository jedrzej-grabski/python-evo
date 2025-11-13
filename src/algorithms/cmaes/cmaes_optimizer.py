import numpy as np
from numpy.typing import NDArray

from src.core.base_optimizer import BaseOptimizer, OptimizationResult
from src.algorithms.cmaes.cmaes_config import CMAESConfig
from src.algorithms.choices import AlgorithmChoice
from src.utils.helpers import norm
from src.utils.boundary_handlers import BoundaryHandler


class CMAESOptimizer(BaseOptimizer):
    def __init__(
        self,
        func,
        initial_point: NDArray[np.float64],
        config: CMAESConfig,
        boundary_handler: BoundaryHandler | None = None,
        boundary_strategy=None,
        lower_bounds=-100.0,
        upper_bounds=100.0,
    ):
        super().__init__(
            func=func,
            initial_point=initial_point,
            config=config,
            algorithm=AlgorithmChoice.CMAES,
            boundary_handler=boundary_handler,
            boundary_strategy=boundary_strategy,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    def optimize(self) -> OptimizationResult:
        cfg = self.config
        N = self.dimensions
        lam = cfg.lambda_
        mu = cfg.mu
        w = cfg.weights
        mueff = (np.sum(w) ** 2) / np.sum(w**2)
        cc = cfg.cc
        cs = cfg.cs
        ccov_mu = cfg.ccov_mu
        ccov_1 = cfg.ccov_1
        damps = cfg.damps

        xmean = self.initial_point.astype(float).copy()
        sigma = float(cfg.sigma)

        pc = np.zeros(N)
        ps = np.zeros(N)
        B = np.eye(N)
        D = np.eye(N)
        BD = B @ D
        C = BD @ BD.T

        chiN = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N**2))

        best_fitness = float("inf")
        best_solution = xmean.copy()
        iteration = 0
        cviol_total = 0

        while self.evaluations < cfg.budget and iteration < cfg.max_iterations:
            iteration += 1

            # Sample new population
            arz = np.random.randn(N, lam)
            arx = xmean[:, None] + sigma * (BD @ arz)
            population = arx.T

            # Boundary handling (non-Lamarckian penalty)
            repaired = np.zeros_like(population)
            pen = np.ones(lam)
            fitness = np.zeros(lam)

            budget_left = cfg.budget - self.evaluations
            eval_count = min(lam, budget_left)
            for i in range(lam):
                x = population[i]
                xr = self.boundary_handler.repair(x)
                repaired[i] = xr
                if not np.array_equal(x, xr):
                    pen[i] = 1 + np.sum((x - xr) ** 2)
                if i < eval_count:
                    fitness[i] = self.func(xr)
                    self.evaluations += 1
                else:
                    fitness[i] = float("inf")

            arfitness = fitness * pen
            valid = pen <= 1
            cviol_total += int(np.sum(pen > 1))

            if np.any(valid):
                idx_valid = np.argmin(fitness[valid])
                global_idx = np.flatnonzero(valid)[idx_valid]
                if fitness[valid][idx_valid] < best_fitness:
                    best_fitness = float(fitness[valid][idx_valid])
                    best_solution = population[global_idx].copy()

            # Sort by penalized fitness
            idx_sorted = np.argsort(arfitness)
            parents = idx_sorted[:mu]

            selx = arx[:, parents]
            xmean = (selx @ w).astype(float)
            selz = arz[:, parents]
            zmean = (selz @ w).astype(float)

            # Cumulation for step-size (ps)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ zmean)

            if cfg.do_hsig:
                hsig = (
                    1
                    if norm(ps)
                    / np.sqrt(1 - (1 - cs) ** (2 * self.evaluations / lam))
                    / chiN
                    < (1.4 + 2 / (N + 1))
                    else 0
                )
            else:
                hsig = 1

            # Cumulation for covariance (pc)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (BD @ zmean)

            # Covariance update
            BDz = BD @ selz
            C = (
                (1 - ccov_1 - ccov_mu) * C
                + ccov_1 * (pc[:, None] @ pc[None, :] + (1 - hsig) * cc * (2 - cc) * C)
                + ccov_mu * BDz @ np.diag(w) @ BDz.T
            )

            # Step-size adaptation
            sigma *= np.exp((norm(ps) / chiN - 1) * cs / damps)

            # Eigen decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.any(eigvals <= 0):
                    if cfg.terminate_cov_mat_cond:
                        msg = "Covariance matrix not positive definite."
                        break
                    eigvals = np.clip(eigvals, 1e-15, None)
                B = eigvecs
                D = np.diag(np.sqrt(eigvals))
                BD = B @ D
                eigenvalues_sorted = np.sort(eigvals)[::-1]
            except np.linalg.LinAlgError:
                eigenvalues_sorted = None
                if cfg.terminate_cov_mat_cond:
                    msg = "Eigen decomposition failed."
                    break

            # Metrics
            finite = arfitness[np.isfinite(arfitness)]
            if finite.size > 0:
                best_iter = float(np.min(finite))
                worst_iter = float(np.max(finite))
                mean_iter = float(np.mean(finite))
            else:
                best_iter = best_fitness
                worst_iter = best_fitness
                mean_iter = best_fitness

            # Flatland escape
            if (
                cfg.flatland_escape
                and finite.size > 0
                and arfitness[idx_sorted[0]]
                == arfitness[idx_sorted[min(1 + lam // 2, 2 + (lam + 3) // 4)]]
            ):
                sigma *= np.exp(0.2 + cs / damps)
                if cfg.trace:
                    print("Flat fitness function. Increasing sigma.")

            # Logging
            self.logger.log_iteration(
                iteration=iteration,
                evaluations=self.evaluations,
                best_fitness=best_iter,
                worst_fitness=worst_iter,
                mean_fitness=mean_iter,
                sigma=sigma,
                fitness=arfitness,
                population=repaired if cfg.diag_pop else None,
                best_solution=best_solution,
                eigenvalues=eigenvalues_sorted,
            )

            # Termination
            if cfg.terminate_stopfitness and best_iter <= cfg.stop_fitness:
                msg = "Stop fitness reached."
                return OptimizationResult(
                    best_solution=best_solution,
                    best_fitness=best_fitness,
                    evaluations=self.evaluations,
                    message=msg,
                    diagnostic=self.get_logs(),
                    algorithm=AlgorithmChoice.CMAES,
                )

            if cfg.terminate_std_dev_tol:
                tolx = cfg.tolx_factor * cfg.sigma
                if np.all(np.diag(D) < tolx) and np.all(sigma * pc < tolx):
                    msg = "Std dev below tolerance."
                    return OptimizationResult(
                        best_solution=best_solution,
                        best_fitness=best_fitness,
                        evaluations=self.evaluations,
                        message=msg,
                        diagnostic=self.get_logs(),
                        algorithm=AlgorithmChoice.CMAES,
                    )

            if cfg.terminate_maxiter and iteration >= cfg.max_iterations:
                msg = "Exceeded maximal iterations."
                return OptimizationResult(
                    best_solution=best_solution,
                    best_fitness=best_fitness,
                    evaluations=self.evaluations,
                    message=msg,
                    diagnostic=self.get_logs(),
                    algorithm=AlgorithmChoice.CMAES,
                )

        msg = "Budget exhausted."
        return OptimizationResult(
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations=self.evaluations,
            message=msg,
            diagnostic=self.get_logs(),
            algorithm=AlgorithmChoice.CMAES,
        )
