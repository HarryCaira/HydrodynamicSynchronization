import numpy as np
from abc import ABC, abstractmethod

from src.hydrosync.constants import GlobalConstants
from src.hydrosync.fluid_dynamics_tensors import FluidDynamicsTensorInterface
from src.hydrosync.force_computation import ForceComputationInterface


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
        n_particles = positions.shape[1]
        tensor = self._fluid_dynamics_tensor.compute_tensor(positions)

        total_forces = np.zeros_like(positions)
        for force_computation in self._external_forces:
            total_forces += force_computation.compute_forces(positions, orbit_centers)

        displacements = np.zeros_like(positions)
        for i in range(n_particles):
            for j in range(n_particles):
                displacements[:, i] += (self._constants.time_step * tensor[i, j] @ total_forces[:, j]) / (self._constants.kB * self._constants.T)
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

        displacements = np.random.multivariate_normal(np.zeros(3 * n_particles), cov).reshape(n_particles, 3).T
        return displacements

    @classmethod
    def create(
        cls,
        constants: GlobalConstants,
        fluid_dynamics_tensor: FluidDynamicsTensorInterface,
        external_forces: list[ForceComputationInterface] = [],
    ) -> "BrownianDisplacement":
        return cls(constants.time_step, fluid_dynamics_tensor)
