import numpy as np
from abc import ABC, abstractmethod

from src.hydrosync.constants import GlobalConstants


class ForceComputationInterface(ABC):
    @abstractmethod
    def compute_forces(self, positions: np.ndarray, orbit_centers: np.ndarray) -> np.ndarray: ...

    @classmethod
    @abstractmethod
    def create(cls, constants: GlobalConstants) -> "ForceComputationInterface": ...


class TangentialDrivingForce(ForceComputationInterface):
    """Calculates driving forces on all particles"""

    def __init__(self, force_strength: float, modulation_weight: float):
        self._force_strength = force_strength
        self._modulation_weight = modulation_weight

    def compute_forces(self, positions: np.ndarray, orbit_centers: np.ndarray) -> np.ndarray:
        """
        Compute tangential driving forces
        Args:
            positions: Current particle positions (3, n)
            orbit_centers: Orbit center positions (3, n)
        Returns:
            Force vector (3, n)
        """
        displacements = positions - orbit_centers
        angles = np.arctan2(displacements[1, :], displacements[0, :])

        forces = np.zeros_like(positions)
        forces[0, :] = self._force_strength * np.sin(angles) * (1 + self._modulation_weight * np.cos(angles))
        forces[1, :] = -self._force_strength * np.cos(angles) * (1 + self._modulation_weight * np.cos(angles))
        return forces

    @classmethod
    def create(cls, constants: GlobalConstants) -> "TangentialDrivingForce":
        return cls(constants.driving_force, constants.driving_force_modulation_strength)


class RadialRestoringForce(ForceComputationInterface):
    """Calculates Hooke's law forces on all particles"""

    def __init__(self, spring_constant: float, radius: float):
        self._spring_constant = spring_constant
        self._orbit_radius = radius

    def compute_forces(self, positions: np.ndarray, orbit_centers: np.ndarray) -> np.ndarray:
        """
        Compute Hooke's law forces
        Args:
            positions: Current particle positions (3, n)
            orbit_centers: Orbit center positions (3, n)
        Returns:
            Force vector (3, n)
        """
        displacements = positions - orbit_centers
        norms = np.linalg.norm(displacements, axis=0)
        radial_unit_vectors = displacements / norms
        extensions = displacements - (radial_unit_vectors * self._orbit_radius)
        forces = -self._spring_constant * extensions
        return forces

    @classmethod
    def create(cls, constants: GlobalConstants) -> "RadialRestoringForce":
        return cls(constants.spring_constant, constants.orbit_radius)
