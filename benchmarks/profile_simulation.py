"""Where does the simulation spend its time, and how does that scale with array size?

Run with:
    uv run python -m benchmarks.profile_simulation
    uv run python -m benchmarks.profile_simulation --tensor-type rotne-prager --max-size 12

The per-step cost of the simulation decomposes into five isolatable pieces. We time
each one on its own across a sweep of (square) array sizes and fit a power law
time ~ n^p to each, so you can see both the absolute breakdown and the asymptotic
scaling exponent. n is the particle count (nx * ny); cost depends on total n, not shape.

Per step the integrator runs HydrodynamicDisplacement then BrownianDisplacement. Both
query the tensor, but the simulation wraps it in a CachedTensor so it is built once per
step and shared, not twice:
  A  compute_tensor      built once per step (shared via CachedTensor)
  B  forces              tangential + radial, vectorised        (hydro only)
  C  contraction         the Python i,j loop tensor[i,j] @ F[j] (hydro only)
  D  covariance build    transpose/reshape to a 3n x 3n matrix  (brownian only)
  E  multivariate_normal sampling from that covariance          (brownian only)

so modelled per-step total = A + B + C + D + E.
"""

import time

import click
import numpy as np

from src.constants import GlobalConstants
from src.arrays import GridArray
from src.fluid_dynamics_tensors import OseenTensor, RotnePragerTensor
from src.force_computation import TangentialDrivingForce, RadialRestoringForce

TENSORS = {"oseen": OseenTensor, "rotne-prager": RotnePragerTensor}


def _time(fn, repeats: int) -> float:
    """Best-of-N wall time for a single call to fn (min reduces scheduler noise)."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def profile_size(size: int, tensor_type: str, repeats: int) -> dict[str, float]:
    """Time each per-step component once, on a representative configuration."""
    constants = GlobalConstants.create()
    array = GridArray.create(nx=size, ny=size, constants=constants, random_start_seed=1)
    tensor = TENSORS[tensor_type].create(constants)
    forces = [TangentialDrivingForce.create(constants), RadialRestoringForce.create(constants)]

    positions = array.positions
    centers = array.orbit_centers
    n = positions.shape[1]
    dt, kT = constants.time_step, constants.kB * constants.T

    # Precompute a tensor so the contraction / covariance / sampling stages are timed
    # in isolation, not bundled with the tensor build.
    T = tensor.compute_tensor(positions)
    cov = 2 * dt * T.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)

    def contraction() -> None:
        total_forces = sum(f.compute_forces(positions, centers) for f in forces)
        disp = np.zeros_like(positions)
        for i in range(n):
            for j in range(n):
                disp[:, i] += (dt * T[i, j] @ total_forces[:, j]) / kT

    a = _time(lambda: tensor.compute_tensor(positions), repeats)
    b = _time(lambda: sum(f.compute_forces(positions, centers) for f in forces), repeats)
    c = _time(contraction, repeats)
    d = _time(lambda: 2 * dt * T.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n), repeats)
    # check_valid='ignore': we only care about timing, not whether the cov is PD here.
    e = _time(lambda: np.random.multivariate_normal(np.zeros(3 * n), cov, check_valid="ignore"), repeats)

    return {"n": n, "tensor": a, "forces": b, "contraction": c, "cov": d, "sampling": e, "per_step": a + b + c + d + e}


def _fit_exponent(ns: list[int], times: list[float]) -> float:
    """Slope of log(time) vs log(n): the power p in time ~ n^p."""
    slope, _ = np.polyfit(np.log(ns), np.log(times), 1)
    return float(slope)


@click.command()
@click.option("-t", "--tensor-type", default="oseen", type=click.Choice(list(TENSORS)), help="Tensor implementation to profile")
@click.option("--max-size", default=10, type=int, help="Largest square grid side length to test (sweep is 2..max-size)")
@click.option("-r", "--repeats", default=5, type=int, help="Best-of-N timing repeats per measurement")
def main(tensor_type: str, max_size: int, repeats: int) -> None:
    sizes = list(range(2, max_size + 1))
    rows = [profile_size(s, tensor_type, repeats) for s in sizes]

    components = ["tensor", "forces", "contraction", "cov", "sampling"]
    header = f"{'n':>5} | " + " | ".join(f"{c:>11}" for c in components) + f" | {'per-step':>11} | {'%tensor':>8}"
    print(f"\nPer-component time per step (ms)  -  tensor='{tensor_type}', best-of-{repeats}\n")
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = " | ".join(f"{r[c] * 1e3:11.4f}" for c in components)
        tensor_share = r["tensor"] / r["per_step"] * 100
        print(f"{r['n']:>5} | {cells} | {r['per_step'] * 1e3:11.4f} | {tensor_share:7.1f}%")

    ns = [r["n"] for r in rows]
    print("\nScaling exponent p  (time ~ n^p):")
    for c in components + ["per_step"]:
        print(f"  {c:>12}: n^{_fit_exponent(ns, [r[c] for r in rows]):.2f}")
    print("\nNote: 'tensor' is built once per step; the simulation shares it across both displacement steps via CachedTensor.")


if __name__ == "__main__":
    main()
