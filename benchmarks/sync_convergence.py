"""Verify that synchronization is a converged physical result, not a timestep artifact.

Two studies, both keyed to the orbital period (the natural clock of the rotors) so time is
reported in dt-independent physical units:

  converge:  run the full noisy dynamics for a fixed number of orbital periods across a
             ladder of dt, reporting the plateau order parameter and the time to lock
             (in orbital periods). A trustworthy regime is one where BOTH are flat as
             dt -> 0. Coarse dt typically gets the plateau right but reports sync as
             happening too fast.

  artifact:  isolate the pure deterministic integration error - run drift only (Brownian
             noise off; with Peclet ~ 1e6 it is negligible) and measure the order parameter
             reached at a fixed early time as dt is swept, then fit error vs dt. Explicit
             Euler is formally first order, but because the motion is rotational its leading
             O(dt) error is oscillatory and averages out per orbit, so the synchronization
             artifact converges at ~O(dt^2).

Run with:
    uv run python -m benchmarks.sync_convergence converge --nx 4 --ny 4 --orbits 20
    uv run python -m benchmarks.sync_convergence artifact --orbits-observed 2
"""

import numpy as np
import click

from src.constants import GlobalConstants
from src.arrays import GridArray
from src.fluid_dynamics_tensors import RotnePragerTensor
from src.force_computation import TangentialDrivingForce, RadialRestoringForce
from src.displacement_computation import HydrodynamicDisplacement, BrownianDisplacement


def _orbital_period(constants: GlobalConstants) -> float:
    """Period of one driven revolution: 2*pi*orbit_radius / (mu*F), with mu = 1/(6 pi eta a)."""
    mobility = 1.0 / (6 * np.pi * constants.eta * constants.a)
    angular_velocity = mobility * constants.driving_force / constants.orbit_radius
    return 2 * np.pi / angular_velocity


def _kuramoto(positions: np.ndarray, centers: np.ndarray) -> float:
    """Global Kuramoto order parameter from the orbital phases (angle about each centre)."""
    angles = np.arctan2(positions[1] - centers[1], positions[0] - centers[0])
    return float(np.abs(np.mean(np.exp(1j * angles))))


def _make_constants(dt: float, steps: int, modulation: float, spring_constant: float, orbit_separation: float) -> GlobalConstants:
    return GlobalConstants.create(
        time_step=dt,
        simulation_steps=steps,
        driving_force_modulation_strength=modulation,
        spring_constant=spring_constant,
        orbit_separation=orbit_separation,
    )


def _rollout(constants: GlobalConstants, nx: int, ny: int, seed: int, with_noise: bool) -> np.ndarray:
    """Integrate the dynamics, returning the per-step Kuramoto order-parameter history.

    Uses a manual loop (not Simulation) to avoid progress-bar output and to allow turning
    the Brownian term off for a clean deterministic artifact measurement.
    """
    np.random.seed(seed)
    array = GridArray.create(nx=nx, ny=ny, constants=constants, random_start_seed=seed)
    tensor = RotnePragerTensor.create(constants)
    displacements = [
        HydrodynamicDisplacement.create(constants, tensor, [TangentialDrivingForce.create(constants), RadialRestoringForce.create(constants)])
    ]
    if with_noise:
        displacements.append(BrownianDisplacement.create(constants, tensor, []))

    positions = array.positions.copy()
    centers = array.orbit_centers
    steps = constants.simulation_steps
    history = np.empty(steps + 1)
    history[0] = _kuramoto(positions, centers)
    for i in range(steps):
        delta = sum(d.compute_displacements(positions, centers) for d in displacements)
        positions = positions + delta
        history[i + 1] = _kuramoto(positions, centers)
    return history


def _time_to_threshold(history: np.ndarray, dt: float, period: float, threshold: float) -> float:
    """Time (in orbital periods) at which the order parameter first reaches `threshold`."""
    if history.max() < threshold:
        return float("nan")
    return float(np.argmax(history >= threshold) * dt / period)


def _fit_power_law(dts: list[float], errors: list[float]) -> float:
    """Slope p of log(error) vs log(dt): the exponent in error ~ dt^p."""
    slope, _ = np.polyfit(np.log(dts), np.log(errors), 1)
    return float(slope)


@click.group()
def cli() -> None:
    """Synchronization convergence and timestep-artifact studies."""


_SHARED = [
    click.option("--nx", default=4, type=int, help="Particles in x"),
    click.option("--ny", default=4, type=int, help="Particles in y"),
    click.option("--modulation", default=0.2, type=float, help="Driving-force modulation (symmetry-breaking that drives sync)"),
    click.option("--spring-constant", default=0.001, type=float, help="Radial spring constant"),
    click.option("--orbit-separation", default=6e-6, type=float, help="Distance between orbit centres (m)"),
]


def _shared_options(fn):
    for option in reversed(_SHARED):
        fn = option(fn)
    return fn


@cli.command()
@_shared_options
@click.option("--orbits", default=20, type=int, help="Number of orbital periods to simulate")
@click.option("--coarsest-dt", default=1e-5, type=float, help="Largest dt; the ladder halves from here")
@click.option("--levels", default=4, type=int, help="Number of dt levels (each half the previous)")
@click.option("--seeds", default=3, type=int, help="Ensemble size")
@click.option("--threshold", default=0.9, type=float, help="Order parameter defining 'synchronized'")
def converge(nx, ny, modulation, spring_constant, orbit_separation, orbits, coarsest_dt, levels, seeds, threshold) -> None:
    """dt-convergence of the plateau order parameter and the time to synchronize."""
    period = _orbital_period(_make_constants(coarsest_dt, 1, modulation, spring_constant, orbit_separation))
    total_time = orbits * period
    dts = [coarsest_dt / 2**level for level in range(levels)]

    print(f"\nconverge: {nx}x{ny} grid, modulation={modulation}, spring={spring_constant}, {orbits} orbital periods, {seeds} seeds")
    print(f"orbital period = {period * 1e3:.2f} ms;  trustworthy = plateau AND sync-time both flat as dt shrinks\n")
    print(f"{'dt':>10} {'steps':>9} {'plateau r':>11} {'sync-time (orbits)':>20}")
    print("-" * 54)
    for dt in dts:
        steps = round(total_time / dt)
        plateaus, sync_times = [], []
        for seed in range(seeds):
            c = _make_constants(dt, steps, modulation, spring_constant, orbit_separation)
            history = _rollout(c, nx, ny, seed, with_noise=True)
            plateaus.append(np.mean(history[int(0.75 * len(history)):]))
            sync_times.append(_time_to_threshold(history, dt, period, threshold))
        print(f"{dt:>10.2e} {steps:>9d} {np.mean(plateaus):>11.3f} {np.nanmean(sync_times):>20.2f}")


@cli.command()
@_shared_options
@click.option("--orbits-observed", default=2.0, type=float, help="Fixed early time (orbital periods) at which to read the artifact")
@click.option("--dt-min", default=1e-6, type=float, help="Smallest dt in the sweep (also the reference)")
@click.option("--dt-max", default=2e-5, type=float, help="Largest dt in the sweep")
@click.option("--n-dt", default=10, type=int, help="Number of dt values (log-spaced)")
@click.option("--seeds", default=3, type=int, help="Ensemble of initial conditions")
def artifact(nx, ny, modulation, spring_constant, orbit_separation, orbits_observed, dt_min, dt_max, n_dt, seeds) -> None:
    """Deterministic integration-artifact scaling: order parameter at a fixed early time vs dt."""
    period = _orbital_period(_make_constants(dt_min, 1, modulation, spring_constant, orbit_separation))
    t_obs = orbits_observed * period
    reference_dt = dt_min / 2
    dts = list(np.geomspace(dt_min, dt_max, n_dt))

    def r_obs(dt: float) -> float:
        steps = round(t_obs / dt)
        c = _make_constants(dt, steps, modulation, spring_constant, orbit_separation)
        return float(np.mean([_rollout(c, nx, ny, seed, with_noise=False)[-1] for seed in range(seeds)]))

    reference = r_obs(reference_dt)
    print(f"\nartifact: {nx}x{ny} grid, DRIFT ONLY, order parameter at t={orbits_observed} orbits, {seeds} inits")
    print(f"reference (dt={reference_dt:.1e}): r = {reference:.4f}\n")
    print(f"{'dt':>10} {'omega*dt':>9} {'r(t_obs)':>10} {'|error|':>10}")
    print("-" * 43)
    omega = 2 * np.pi / period
    fit_dts, fit_errs = [], []
    for dt in dts:
        r = r_obs(dt)
        err = abs(r - reference)
        print(f"{dt:>10.2e} {omega * dt:>9.4f} {r:>10.4f} {err:>10.4f}")
        if err > 1e-3 and dt <= dt_max / 3:  # asymptotic band: above noise floor, below saturation
            fit_dts.append(dt)
            fit_errs.append(err)
    if len(fit_dts) >= 2:
        p = _fit_power_law(fit_dts, fit_errs)
        print(f"\nasymptotic scaling: artifact ~ dt^{p:.2f}  (rotational motion => leading O(dt) cancels, expect ~2)")
    else:
        print("\nnot enough points above the noise floor to fit; widen the dt range")


if __name__ == "__main__":
    cli()
