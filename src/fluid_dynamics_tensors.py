import numpy as np
from abc import ABC, abstractmethod

from src.constants import GlobalConstants


class FluidDynamicsTensorInterface(ABC):
    @abstractmethod
    def compute_tensor(self, positions: np.ndarray) -> np.ndarray: ...

    @classmethod
    @abstractmethod
    def create(cls, constants: GlobalConstants) -> "FluidDynamicsTensorInterface": ...


class OseenTensor(FluidDynamicsTensorInterface):
    """
    Calculates fluid interactions between particles using the Oseen tensor formalism.
    The Oseen tensor describes how the motion of one particle affects another through the fluid.
    """

    def __init__(self, constants: GlobalConstants, prefactor: float) -> None:
        self._constants = constants
        self._prefactor = prefactor

    def _self_mobility(self) -> np.ndarray:
        """Calculate self-mobility tensor (diagonal elements)"""
        mobility: float = self._prefactor / (6 * np.pi * self._constants.eta * self._constants.a)
        return mobility * np.identity(3)

    def _cross_mobility(self, r_ij: np.ndarray) -> np.ndarray:
        """
        Calculate cross-mobility tensor (off-diagonal elements)
        Args:
            r_ij: Vector between particles i and j
        """
        r: float = float(np.linalg.norm(r_ij))
        if r < 2 * self._constants.a:  # Prevent overlap
            r = 2 * self._constants.a

        prefactor: float = self._prefactor / (8 * np.pi * self._constants.eta * r)
        r_hat: np.ndarray = r_ij / r

        return prefactor * (np.identity(3) + np.outer(r_hat, r_hat))

    def compute_tensor(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute the full Oseen tensor for all particle pairs
        Args:
            positions: Array of shape (3, n) containing particle positions
        Returns:
            Oseen tensor of shape (n, n, 3, 3)
        """
        n_particles = positions.shape[1]
        tensor = np.zeros((n_particles, n_particles, 3, 3))

        for i in range(n_particles):
            tensor[i, i] = self._self_mobility()
            for j in range(i + 1, n_particles):
                r_ij = positions[:, i] - positions[:, j]
                tensor[i, j] = tensor[j, i] = self._cross_mobility(r_ij)
        return tensor

    @classmethod
    def create(cls, constants: GlobalConstants) -> "OseenTensor":
        return cls(constants=constants, prefactor=constants.kB * constants.T)


class RotnePragerTensor(FluidDynamicsTensorInterface):
    """
    Calculates fluid interactions between particles using the Rotne-Prager-Yamakawa
    tensor formalism.

    Unlike the Oseen tensor, the Rotne-Prager-Yamakawa tensor accounts for the finite
    size of the particles (corrections of order (a/r)^2) and stays positive-definite
    even when particles overlap, by switching to a regularised form for separations
    smaller than 2a. The two branches are continuous at r = 2a.
    """

    def __init__(self, constants: GlobalConstants, prefactor: float) -> None:
        self._constants = constants
        self._prefactor = prefactor

    def _self_mobility(self) -> np.ndarray:
        """Calculate self-mobility tensor (diagonal elements)"""
        mobility: float = self._prefactor / (6 * np.pi * self._constants.eta * self._constants.a)
        return mobility * np.identity(3)

    def _cross_mobility(self, r_ij: np.ndarray) -> np.ndarray:
        """
        Calculate cross-mobility tensor (off-diagonal elements)
        Args:
            r_ij: Vector between particles i and j
        """
        a: float = self._constants.a
        r: float = float(np.linalg.norm(r_ij))
        r_hat: np.ndarray = r_ij / r if r > 0 else np.zeros(3)
        outer: np.ndarray = np.outer(r_hat, r_hat)

        if r >= 2 * a:
            prefactor: float = self._prefactor / (8 * np.pi * self._constants.eta * r)
            return prefactor * ((1 + 2 * a**2 / (3 * r**2)) * np.identity(3) + (1 - 2 * a**2 / r**2) * outer)

        # Overlapping spheres: regularised Rotne-Prager-Yamakawa form
        prefactor = self._prefactor / (6 * np.pi * self._constants.eta * a)
        return prefactor * ((1 - 9 * r / (32 * a)) * np.identity(3) + (3 * r / (32 * a)) * outer)

    def compute_tensor(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute the full Rotne-Prager tensor for all particle pairs
        Args:
            positions: Array of shape (3, n) containing particle positions
        Returns:
            Rotne-Prager tensor of shape (n, n, 3, 3)
        """
        n_particles = positions.shape[1]
        tensor = np.zeros((n_particles, n_particles, 3, 3))

        for i in range(n_particles):
            tensor[i, i] = self._self_mobility()
            for j in range(i + 1, n_particles):
                r_ij = positions[:, i] - positions[:, j]
                tensor[i, j] = tensor[j, i] = self._cross_mobility(r_ij)
        return tensor

    @classmethod
    def create(cls, constants: GlobalConstants) -> "RotnePragerTensor":
        return cls(constants=constants, prefactor=constants.kB * constants.T)
