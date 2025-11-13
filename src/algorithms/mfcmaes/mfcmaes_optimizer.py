from __future__ import annotations
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from src.core.base_optimizer import BaseOptimizer, OptimizationResult
from src.algorithms.choices import AlgorithmChoice
from src.algorithms.mfcmaes.mfcmaes_config import MFCMAESConfig
from src.utils.helpers import norm
from src.utils.boundary_handlers import BoundaryHandler


class MFCMAESOptimizer(BaseOptimizer, tuple[()]):
    def __init__(
        self,
        func,
        initial_point: NDArray[np.float64],
        config: MFCMAESConfig,
        boundary_handler: BoundaryHandler | None = None,
        boundary_strategy=None,
        lower_bounds=-100.0,
        upper_bounds=100.0,
    ):
        super().__init__(
            func=func,
            initial_point=initial_point,
            config=config,
            algorithm=AlgorithmChoice.MFCMAES,
            boundary_handler=boundary_handler,
            boundary_strategy=boundary_strategy,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        if config.seed is not None:
            np.random.seed(config.seed)

    def _init_strategy(self) -> dict:
        N = self.dimensions
        lam = self.config.population_size
        mu = self.config.mu if self.config.mu is not None else int(np.floor(lam / 2))

        if self.config.equal_weights:
            weights = np.ones(mu) / mu
        else:
            # Optional: log weights (not used in provided R code)
            ranks = np.arange(1, mu + 1)
            weights = np.log(mu + 0.5) - np.log(ranks)
            weights = weights / np.sum(weights)

        mueff = (np.sum(weights) ** 2) / np.sum(weights**2)
        cc = 4.0 / (N + 4.0)
        cs = (mueff + 2.0) / (N + mueff + 3.0)
        mucov = mueff
        ccov = (1.0 / mucov) * 2.0 / (N + 1.4) ** 2 + (1.0 - 1.0 / mucov) * (
            (2.0 * mucov - 1.0) / ((N + 2.0) ** 2 + 2.0 * mucov)
        )
        c_mu = ccov * (1.0 - 1.0 / mucov)
        c_1 = ccov - c_mu
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (N + 1.0)) - 1.0) + cs

        window = int(self.config.window)
        t_idx = np.arange(1, max(2, self.config.max_iterations) + 1)
        decay_table = (1.0 - ccov) ** ((t_idx - 1.0) / 2.0)

        return {
            "N": N,
            "lambda": lam,
            "mu": mu,
            "weights": weights,
            "mueff": mueff,
            "cc": cc,
            "cs": cs,
            "mucov": mucov,
            "ccov": ccov,
            "c_mu": c_mu,
            "c_1": c_1,
            "damps": damps,
            "window": window,
            "decay_table": decay_table,
        }

    @staticmethod
    def _rotate_right(v: NDArray[np.float64], n: int) -> NDArray[np.float64]:
        if len(v) == 0:
            return v
        n = n % len(v)
        if n == 0:
            return v
        return np.concatenate([v[-n:], v[:-n]])

    def _generate_population(
        self,
        xmean: NDArray[np.float64],
        sigma: float,
        state: dict,
        d_history: NDArray[np.float64],  # shape (N, window*mu)
        p_history: NDArray[np.float64],  # shape (N, window)
        t: int,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        N = state["N"]
        lam = state["lambda"]
        mu = state["mu"]
        c_mu = state["c_mu"]
        c_1 = state["c_1"]
        ccov = state["ccov"]
        window = state["window"]
        decay_table = state["decay_table"]
        weights = state["weights"]

        # Prepare time-decay vector for last `window` generations
        base_decay = decay_table[:window][::-1]  # same as R: decay_table[window:1]
        decay = self._rotate_right(base_decay, t - 1)  # align with current t

        # Scale vectors for rank-mu component
        decay_rep = np.repeat(decay, mu)  # length window*mu
        w = np.tile(
            np.sqrt(weights), window
        )  # length window*mu (match d_history column layout)

        # Random mixes
        r_mu = np.random.randn(window * mu, lam)  # (window*mu) x lam
        inner_sum = d_history @ (r_mu * (decay_rep * w)[:, None])  # N x lam
        rank_mu = np.sqrt(c_mu) * inner_sum

        r_1 = np.random.randn(window, lam)  # window x lam
        rank_1 = np.sqrt(c_1) * (p_history @ (r_1 * decay[:, None]))  # N x lam

        outer_sum = rank_mu + rank_1

        # Tail contribution
        if t <= window:
            last_decay = decay_table[t - 1]  # t in [1..window]
        else:
            last_decay = np.sqrt(
                (1.0 - ccov) ** (t - 1)
                + (1.0 - ccov) ** window * (1.0 - ccov) ** (t - window - 1)
            )

        r_last = np.random.randn(N, lam)
        last_term = last_decay * r_last

        d = outer_sum + last_term  # N x lam
        arx = xmean[:, None] + sigma * d  # N x lam

        return arx, d

    def optimize(self) -> OptimizationResult:
        cfg = self.config
        st = self._init_strategy()
        N, lam, mu = st["N"], st["lambda"], st["mu"]
        weights = st["weights"]
        mueff, cc, cs, damps = st["mueff"], st["cc"], st["cs"], st["damps"]

        xmean = self.initial_point.astype(float).copy()
        sigma = float(cfg.sigma)

        # Archives
        window = st["window"]
        d_history = np.zeros((N, window * mu))
        p_history = np.zeros((N, window))
        pc = np.zeros(N)

        iteration = 0
        best_fitness = float("inf")
        best_solution = xmean.copy()
        cviol_total = 0

        # For PPMF
        prev_midpoint_fitness = float("inf")
        midpoint_fitness = float("inf")

        # Convenience index helpers
        def p_index(t: int) -> int:
            return (t - 1) % window

        def d_range(t: int) -> slice:
            start = p_index(t) * mu
            end = start + mu
            return slice(start, end)

        # Main loop
        while self.evaluations < cfg.budget and iteration < cfg.max_iterations:
            iteration += 1
            # Sample population (matrix-free)
            arx, d = self._generate_population(
                xmean, sigma, st, d_history, p_history, iteration
            )
            # Convert to (lambda, N) for evaluation and boundary handling
            population = arx.T  # lam x N

            # Repair to bounds and evaluate with penalty (non-Lamarckian, like the R code)
            repaired = np.zeros_like(population)
            fitness = np.zeros(lam)
            pen = np.ones(lam)
            cviol_iter = 0

            budget_left = cfg.budget - self.evaluations
            eval_count = min(lam, budget_left)

            for i in range(lam):
                x = population[i]
                xr = self.boundary_handler.repair(x)
                repaired[i] = xr
                if not np.array_equal(x, xr):
                    cviol_iter += 1
                    # penalty 1 + squared distance
                    pen[i] = 1.0 + np.sum((x - xr) ** 2)
                else:
                    pen[i] = 1.0

                if i < eval_count:
                    fitness[i] = self.func(xr)
                    self.evaluations += 1
                else:
                    fitness[i] = float("inf")

            cviol_total += cviol_iter
            arfitness = fitness * pen

            # Track best among feasible (pen == 1)
            feasible = pen <= 1.0
            if np.any(feasible):
                idx = np.argmin(fitness[feasible])
                fit_candidate = fitness[feasible][idx]
                # best feasible original point equals repaired (no change), pick original in-bounds
                idx_global = np.flatnonzero(feasible)[idx]
                if fit_candidate < best_fitness:
                    best_fitness = float(fit_candidate)
                    best_solution = population[idx_global].copy()

            # Selection
            order = np.argsort(arfitness)
            parents = order[:mu]

            selx = arx[:, parents]  # N x mu
            xmean = (selx @ weights).astype(float)  # N

            seld = d[:, parents]  # N x mu
            dmean = (seld @ weights).astype(float)  # N

            # Cumulation path
            pc = (1.0 - cc) * pc + np.sqrt(cc * (2.0 - cc) * mueff) * dmean

            # Update archives
            d_history[:, d_range(iteration)] = d[:, parents]
            p_history[:, p_index(iteration)] = pc

            # PPMF step-size update (uses repaired mid-point)
            prev_midpoint_fitness = midpoint_fitness
            mean_point = (
                np.mean(repaired[:eval_count], axis=0)
                if eval_count > 0
                else xmean.copy()
            )
            # Evaluate midpoint if budget allows
            if self.evaluations < cfg.budget:
                midpoint_fitness = self.func(mean_point)
                self.evaluations += 1
            else:
                midpoint_fitness = prev_midpoint_fitness

            if np.isfinite(prev_midpoint_fitness):
                p_succ = float(np.sum(arfitness < prev_midpoint_fitness)) / float(lam)
            else:
                p_succ = 0.0  # first iteration fallback

            sigma *= np.exp(
                cfg.damps_ppmf
                * (p_succ - cfg.p_target_ppmf)
                / max(1e-12, (1.0 - cfg.p_target_ppmf))
            )

            # Optional empirical eigen logging
            eigenvalues = None
            if self.config.diag_eigen:
                # Empirical covariance of generated population (columns are samples)
                cov = np.cov(arx)  # rows are variables by default for N x lam
                try:
                    ev = np.linalg.eigvalsh(cov)
                    eigenvalues = np.sort(ev)[::-1]
                except np.linalg.LinAlgError:
                    eigenvalues = None

            # Compute stats for logging
            finite_fit = arfitness[np.isfinite(arfitness)]
            if finite_fit.size > 0:
                best_fit_iter = float(np.min(finite_fit))
                worst_fit_iter = float(np.max(finite_fit))
                mean_fit_iter = float(np.mean(finite_fit))
            else:
                best_fit_iter = best_fitness
                worst_fit_iter = best_fitness
                mean_fit_iter = best_fitness

            # Flatland escape hack (matches your R condition)
            if cfg.flatland_escape and finite_fit.size > 0:
                half = max(1, int(np.floor(lam / 2)))
                qtr = max(1, int(np.ceil(lam / 4)))
                compare_idx = min(1 + half, 2 + qtr) - 1  # zero-based
                if (
                    compare_idx < len(order)
                    and arfitness[order[0]] == arfitness[order[compare_idx]]
                ):
                    sigma *= np.exp(0.2 + cs / damps)
                    if cfg.trace:
                        print("Flat fitness function. Increasing sigma.")

            # Log this iteration
            self.logger.log_iteration(
                iteration=iteration,
                evaluations=self.evaluations,
                best_fitness=best_fit_iter,
                worst_fitness=worst_fit_iter,
                mean_fitness=mean_fit_iter,
                sigma=sigma,
                p_succ=p_succ,
                midpoint_fitness=float(midpoint_fitness),
                constraint_violations=int(cviol_iter),
                fitness=arfitness,
                population=repaired if self.config.diag_pop else None,
                best_solution=best_solution,
                eigenvalues=eigenvalues,
            )

            # Termination by target fitness (on evaluated individuals)
            if best_fit_iter <= cfg.stop_fitness:
                msg = "Stop fitness reached."
                return OptimizationResult(
                    best_solution=best_solution,
                    best_fitness=float(best_fitness),
                    evaluations=self.evaluations,
                    message=msg,
                    diagnostic=self.get_logs(),
                    algorithm=AlgorithmChoice.MFCMAES,
                )

        msg = "Exceeded maximal iterations or budget."
        return OptimizationResult(
            best_solution=best_solution,
            best_fitness=float(best_fitness),
            evaluations=self.evaluations,
            message=msg,
            diagnostic=self.get_logs(),
            algorithm=AlgorithmChoice.MFCMAES,
        )
