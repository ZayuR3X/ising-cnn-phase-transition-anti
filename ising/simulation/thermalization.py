"""
ising/simulation/thermalization.py

Thermalization (burn-in) for the 2D Ising Model CA simulation.
Runs sweeps until magnetization converges, with extended burn-in near Tc.
"""

import numpy as np
from ising.simulation.ca_ising import IsingCA


TC_J1 = 2.2692   # Onsager Tc for J=1


def thermalize(
    ca: IsingCA,
    T: float,
    n_sweeps_default: int = 2_000,
    n_sweeps_critical: int = 8_000,
    critical_window: float = 0.3,
    convergence_eps: float = 0.005,
    check_interval: int = 200,
    verbose: bool = False,
) -> dict:
    """
    Run burn-in sweeps until magnetization converges.

    Runs in blocks of `check_interval` sweeps. Convergence is declared when
    |delta_m| < convergence_eps for two consecutive blocks.
    Hard cap at n_sweeps_default (n_sweeps_critical near Tc).
    """
    beta = 1.0 / T
    Tc_eff = (2.0 * ca.J) / np.log(1.0 + np.sqrt(2.0))

    near_critical = abs(T - Tc_eff) < critical_window
    n_max = n_sweeps_critical if near_critical else n_sweeps_default

    n_sweeps_run = 0
    converged = False
    prev_mean_m = None
    n_converged_checks = 0

    while n_sweeps_run < n_max:
        block_m = []
        for _ in range(check_interval):
            ca.sweep(beta)
            block_m.append(ca.magnetization())
        n_sweeps_run += check_interval

        current_mean_m = float(np.mean(block_m))

        if prev_mean_m is not None:
            delta = abs(current_mean_m - prev_mean_m)
            if delta < convergence_eps:
                n_converged_checks += 1
                if n_converged_checks >= 2:
                    converged = True
                    break
            else:
                n_converged_checks = 0

        prev_mean_m = current_mean_m

    return {
        "n_sweeps_run": n_sweeps_run,
        "converged": converged,
        "final_m": ca.magnetization(),
    }


def smart_init(ca: IsingCA, T: float) -> None:
    """Hot start above Tc, cold start below Tc."""
    Tc_eff = (2.0 * ca.J) / np.log(1.0 + np.sqrt(2.0))
    if T >= Tc_eff - 0.1:
        ca.init_random()
    else:
        sign = 1 if ca.rng.random() > 0.5 else -1
        ca.init_ordered(sign=sign)
