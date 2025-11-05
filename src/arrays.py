import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

from src.constants import GlobalConstants


class ParticleArray2DInterface(ABC):
    @abstractmethod
    def __init__(self, orbit_centers: np.array, positions: np.array) -> None: ...

    @abstractmethod
    def update_positions(self, new_positions: np.array) -> None: ...

    @property
    @abstractmethod
    def positions(self) -> np.array: ...

    @property
    @abstractmethod
    def orbit_centers(self) -> np.array: ...

    @classmethod
    @abstractmethod
    def create(cls, nx: int, ny: int, constants: GlobalConstants, random_seed: Optional[int] = None) -> "ParticleArray2DInterface": ...


class ParticleArray2D(ParticleArray2DInterface):
    def __init__(self, orbit_centers: np.array, positions: np.array) -> None:
        self._orbit_centers = orbit_centers
        self._positions = positions

    def update_positions(self, new_positions: np.array) -> None:
        self._positions = new_positions

    @property
    def positions(self) -> np.array:
        return self._positions

    @property
    def orbit_centers(self) -> np.array:
        return self._orbit_centers

    @classmethod
    def create(cls, nx: int, ny: int, constants: GlobalConstants, random_seed: Optional[int] = None) -> "ParticleArray2D":
        """Create a 2D array of particles in a grid formation"""
        orbit_centers = []
        positions = []
        phases = []

        if random_seed is not None:
            phases = np.random.uniform(0, 2 * np.pi, size=nx * ny)
        else:
            phases = np.zeros(nx * ny)

        for i in range(nx):
            for j in range(ny):
                orbit_center = np.array([i * constants.orbit_separation, j * constants.orbit_separation, 0.0])
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
