import numpy as np
from abc import ABC, abstractmethod

from src.constants import GlobalConstants
from src.fluid_dynamics_tensors import FluidDynamicsTensorInterface
from src.force_computation import ForceComputationInterface


class ParticleDisplacementInterface(ABC):
    @abstractmethod
    def compute_displacements(self, positions: np.ndarray, orbit_centers: np.ndarray) -> np.ndarray: ...

    @classmethod
    @abstractmethod
    def create(
        cls,
        constants: GlobalConstants,
        fluid_dynamics_tensor: FluidDynamicsTensorInterface,
        external_forces: list[ForceComputationInterface],
    ) -> "ParticleDisplacementInterface": ...


class HydrodynamicDisplacement(ParticleDisplacementInterface):
    """Calculates hydrodynamic displacements due to forces"""

    def __init__(
        self,
        constants: GlobalConstants,
        fluid_dynamics_tensor: FluidDynamicsTensorInterface,
        external_forces: list[ForceComputationInterface],
    ):
        self._constants = constants
        self._fluid_dynamics_tensor = fluid_dynamics_tensor
        self._external_forces = external_forces

    def compute_displacements(self, positions: np.ndarray, orbit_centers: np.ndarray) -> np.ndarray:
        """
        Compute hydrodynamic displacements due to forces
        Args:
            positions: Current particle positions (3, n)
            orbit_centers: Orbit center positions (3, n)
        Returns:
            Displacement vector (3, n)
        """
        tensor = self._fluid_dynamics_tensor.compute_tensor(positions)

        total_forces = np.zeros_like(positions)
        for force_computation in self._external_forces:
            total_forces += force_computation.compute_forces(positions, orbit_centers)

        # Deterministic drift v_i = (dt / kT) * sum_j D_ij @ F_j, contracting over the
        # particle index j and the spatial index b. einsum runs the whole contraction in
        # one compiled routine instead of an n^2 Python loop.
        drift = np.einsum("ijab,bj->ai", tensor, total_forces)
        displacements = (self._constants.time_step / (self._constants.kB * self._constants.T)) * drift
        return displacements

    @classmethod
    def create(
        cls,
        constants: GlobalConstants,
        fluid_dynamics_tensor: FluidDynamicsTensorInterface,
        external_forces: list[ForceComputationInterface],
    ) -> "HydrodynamicDisplacement":
        return cls(constants, fluid_dynamics_tensor, external_forces)


class BrownianDisplacement(ParticleDisplacementInterface):
    """Computes displacement due to Brownian motion"""

    def __init__(self, time_step: float, fluid_dynamics_tensor: FluidDynamicsTensorInterface):
        self._dt = time_step
        self._fluid_dynamics_tensor = fluid_dynamics_tensor

    def compute_displacements(self, positions: np.ndarray, orbit_centers: np.ndarray) -> np.ndarray:
        """
        Compute correlated random displacements
        Args:
            positions: Current particle positions (3, n)
        Returns:
            Random displacement vector (3, n)
        """
        n_particles = positions.shape[1]
        tensor = self._fluid_dynamics_tensor.compute_tensor(positions)
        cov = 2 * self._dt * tensor.transpose(0, 2, 1, 3).reshape(3 * n_particles, 3 * n_particles)

        # Sample N(0, cov) via the symmetric eigendecomposition cov = Q diag(w) Q^T.
        # Q diag(sqrt(w)) is a valid square-root factor, so x = Q (sqrt(w) * z) with
        # z ~ N(0, I) has covariance cov. eigh is cheaper than the SVD that
        # np.random.multivariate_normal uses by default, and clipping negative eigenvalues
        # keeps it robust to a non-positive-definite Oseen covariance (nearest PSD sample).
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        sqrt_eigenvalues = np.sqrt(np.clip(eigenvalues, 0.0, None))
        standard_normal = np.random.standard_normal(3 * n_particles)
        sample = eigenvectors @ (sqrt_eigenvalues * standard_normal)

        displacements = sample.reshape(n_particles, 3).T
        return displacements

    @classmethod
    def create(
        cls,
        constants: GlobalConstants,
        fluid_dynamics_tensor: FluidDynamicsTensorInterface,
        external_forces: list[ForceComputationInterface] = [],
    ) -> "BrownianDisplacement":
        return cls(constants.time_step, fluid_dynamics_tensor)
