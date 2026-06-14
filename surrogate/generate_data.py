"""Step 1 - turn the numerical simulator into a labelled dataset.

The numerical simulator is the ground-truth *oracle*. For every step we store the
inputs the surrogate will see (particle positions, the fixed orbit-centre geometry,
and the physical parameters) and the label it must learn: the deterministic drift
v = mu @ F that HydrodynamicDisplacement computes *before* thermal noise is added.

Two design decisions worth flagging:
  - We regress the drift, not the full displacement. The drift is a deterministic
    function of the configuration -- the learnable operator. The Brownian term is
    noise we add back analytically at roll-out, so there is nothing to learn there.
  - We still integrate the *true noisy* dynamics, so the configurations we record
    are the ones the surrogate will actually meet at roll-out (no distribution
    shift). This is why noise stays on even though we never regress it.

The train/test split is by parameter set, never by frame: neighbouring frames of a
trajectory are nearly identical, so a random frame split would leak the test set.
"""

import os
import time
from dataclasses import dataclass

import numpy as np

from src.constants import GlobalConstants
from src.arrays import GridArray
from src.fluid_dynamics_tensors import OseenTensor
from src.force_computation import TangentialDrivingForce, RadialRestoringForce
from src.displacement_computation import HydrodynamicDisplacement, BrownianDisplacement

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@dataclass(frozen=True)
class RolloutConfig:
    driving_force: float
    modulation: float
    seed: int
    nx: int = 3
    ny: int = 3
    spring_constant: float = 0.001
    steps: int = 600


def kuramoto(positions: np.ndarray, centers: np.ndarray) -> float:
    """Phase-coherence order parameter r in [0, 1] for one (3, N) snapshot."""
    angles = np.arctan2(positions[1] - centers[1], positions[0] - centers[0])
    return float(np.abs(np.mean(np.exp(1j * angles))))


def generate_rollout(cfg: RolloutConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    constants = GlobalConstants.create(
        simulation_steps=cfg.steps,
        driving_force=cfg.driving_force,
        driving_force_modulation_strength=cfg.modulation,
        spring_constant=cfg.spring_constant,
    )
    # GridArray's seed argument is really an on/off flag, so seed the global RNG
    # here to make both the initial phases and the noise draws reproducible.
    np.random.seed(cfg.seed)
    array = GridArray.create(nx=cfg.nx, ny=cfg.ny, constants=constants, random_start_seed=cfg.seed)
    tensor = OseenTensor.create(constants)
    drift_model = HydrodynamicDisplacement.create(
        constants,
        tensor,
        [TangentialDrivingForce.create(constants), RadialRestoringForce.create(constants)],
    )
    noise_model = BrownianDisplacement.create(constants, tensor)

    centers = array.orbit_centers
    positions = array.positions
    n = cfg.nx * cfg.ny
    states = np.empty((cfg.steps, 3, n))
    drifts = np.empty((cfg.steps, 3, n))

    for t in range(cfg.steps):
        drift = drift_model.compute_displacements(positions, centers)  # the LABEL
        noise = noise_model.compute_displacements(positions, centers)  # true dynamics keep noise on
        states[t] = positions
        drifts[t] = drift
        positions = positions + drift + noise

    return states, drifts, centers


def build(configs: list[RolloutConfig]) -> dict[str, np.ndarray]:
    states, drifts, centers, params = [], [], [], []
    for cfg in configs:
        s, d, c = generate_rollout(cfg)
        states.append(s)
        drifts.append(d)
        centers.append(c)
        params.append([cfg.driving_force, cfg.modulation, cfg.spring_constant, cfg.seed])
    return {
        "states": np.stack(states),  # (rollouts, steps, 3, N)
        "drifts": np.stack(drifts),  # (rollouts, steps, 3, N)  <- labels
        "centers": np.stack(centers),  # (rollouts, 3, N)
        "params": np.array(params),  # (rollouts, 4): driving_force, modulation, spring_k, seed
    }


# Held-out test uses driving forces *between* the trained values (interpolation)
# and fresh seeds, so "unseen" means genuinely unseen parameters and noise.
TRAIN = [RolloutConfig(driving_force=f, modulation=m, seed=s) for f in (4.5e-9, 5.5e-9, 6.5e-9) for m in (0.1, 0.3) for s in (0, 1, 2)]
TEST = [RolloutConfig(driving_force=f, modulation=m, seed=s) for f in (5.0e-9, 6.0e-9) for m in (0.2,) for s in (7, 8)]


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, configs in (("train", TRAIN), ("test", TEST)):
        start = time.time()
        data = build(configs)
        np.savez(os.path.join(DATA_DIR, f"{name}.npz"), **data)

        r, steps, _, n = data["states"].shape
        sample = data["states"][0]
        c0 = data["centers"][0]
        print(f"\n[{name}] {r} rollouts x {steps} steps x {n} particles = {r * steps * n:,} per-particle samples  ({time.time() - start:.1f}s)")
        print(f"  states {data['states'].shape}  drifts {data['drifts'].shape}")
        print(f"  example rollout: kuramoto r  {kuramoto(sample[0], c0):.3f} (start) -> {kuramoto(sample[-1], c0):.3f} (end)")
        print(f"  drift magnitude: mean |v*dt| = {np.abs(data['drifts']).mean():.2e} m/step (tiny -> we will non-dimensionalise in step 2)")


if __name__ == "__main__":
    main()
