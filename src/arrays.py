import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

from src.constants import GlobalConstants


class ParticleArray2DInterface(ABC):
    @abstractmethod
    def __init__(self, orbit_centers: np.ndarray, positions: np.ndarray) -> None: ...

    @abstractmethod
    def update_positions(self, new_positions: np.ndarray) -> None: ...

    @property
    @abstractmethod
    def positions(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def orbit_centers(self) -> np.ndarray: ...

    @classmethod
    @abstractmethod
    def create(
        cls,
        nx: int,
        ny: int,
        constants: GlobalConstants,
        random_seed: Optional[int] = None,
    ) -> "ParticleArray2DInterface": ...


class GridArray(ParticleArray2DInterface):
    """
    Represents a 2D array of particles arranged in a grid formation.
    Args:
        orbit_centers: Array of shape (3, n) containing the orbit centers of the particles
        positions: Array of shape (3, n) containing the current positions of the particles
    """

    def __init__(self, orbit_centers: np.ndarray, positions: np.ndarray) -> None:
        self._orbit_centers = orbit_centers
        self._positions = positions

    def update_positions(self, new_positions: np.ndarray) -> None:
        self._positions = new_positions

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def orbit_centers(self) -> np.ndarray:
        return self._orbit_centers

    @classmethod
    def create(
        cls,
        nx: int,
        ny: int,
        constants: GlobalConstants,
        random_start_seed: Optional[int] = None,
    ) -> "GridArray":
        orbit_centers = []
        positions = []
        phases = np.array([])

        if random_start_seed is not None:
            phases = np.random.uniform(0, 2 * np.pi, size=nx * ny)
        else:
            phases = np.zeros(nx * ny)

        for i in range(nx):
            for j in range(ny):
                orbit_center = np.array(
                    [
                        i * constants.orbit_separation,
                        j * constants.orbit_separation,
                        0.0,
                    ]
                )
                orbit_centers.append(orbit_center)
                phi = phases[i * ny + j]

                radius = np.array(
                    [
                        constants.orbit_radius * np.cos(phi),
                        constants.orbit_radius * np.sin(phi),
                        0.0,
                    ]
                )
                particle_position = orbit_center + radius
                positions.append(particle_position)

        return cls(np.array(orbit_centers).T, np.array(positions).T)


class HoneycombArray(ParticleArray2DInterface):
    """
    Represents a 2D array of particles arranged in a honeycomb formation.
    Args:
        orbit_centers: Array of shape (3, n) containing the orbit centers of the particles
        positions: Array of shape (3, n) containing the current positions of the particles
    """

    def __init__(self, orbit_centers: np.ndarray, positions: np.ndarray) -> None:
        self._orbit_centers = orbit_centers
        self._positions = positions

    def update_positions(self, new_positions: np.ndarray) -> None:
        self._positions = new_positions

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def orbit_centers(self) -> np.ndarray:
        return self._orbit_centers

    @classmethod
    def create(
        cls,
        nx: int,
        ny: int,
        constants: GlobalConstants,
        random_start_seed: Optional[int] = None,
    ) -> "HoneycombArray":
        orbit_centers = []
        positions = []
        phases = np.array([])

        if random_start_seed is not None:
            phases = np.random.uniform(0, 2 * np.pi, size=nx * ny)
        else:
            phases = np.zeros(nx * ny)

        for i in range(nx):
            for j in range(ny):
                x_offset = (j % 2) * (constants.orbit_separation / 2)
                orbit_center = np.array(
                    [
                        i * constants.orbit_separation + x_offset,
                        j * (constants.orbit_separation * np.sqrt(3) / 2),
                        0.0,
                    ]
                )
                orbit_centers.append(orbit_center)
                phi = phases[i * ny + j]

                radius = np.array(
                    [
                        constants.orbit_radius * np.cos(phi),
                        constants.orbit_radius * np.sin(phi),
                        0.0,
                    ]
                )
                particle_position = orbit_center + radius
                positions.append(particle_position)

        return cls(np.array(orbit_centers).T, np.array(positions).T)
