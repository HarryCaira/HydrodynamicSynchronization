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

    def compute_tensor(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute the full Oseen tensor for all particle pairs (vectorised over pairs).
        Args:
            positions: Array of shape (3, n) containing particle positions
        Returns:
            Oseen tensor of shape (n, n, 3, 3)
        """
        n_particles = positions.shape[1]
        a = self._constants.a

        pos = positions.T  # (n, 3)
        r_vec = pos[:, None, :] - pos[None, :, :]  # (n, n, 3): r_vec[i, j] = pos_i - pos_j
        r = np.linalg.norm(r_vec, axis=2)  # (n, n)
        r_clamped = np.where(r < 2 * a, 2 * a, r)  # prevent overlap (and div-by-zero on the diagonal)

        r_hat = r_vec / r_clamped[:, :, None]  # (n, n, 3); zero on the diagonal
        outer = r_hat[:, :, :, None] * r_hat[:, :, None, :]  # (n, n, 3, 3)

        identity = np.identity(3)
        cross_prefactor = self._prefactor / (8 * np.pi * self._constants.eta * r_clamped)  # (n, n)
        tensor = cross_prefactor[:, :, None, None] * (identity + outer)

        # Overwrite the diagonal with the self-mobility tensor.
        self_mobility = self._prefactor / (6 * np.pi * self._constants.eta * a)
        diagonal = np.arange(n_particles)
        tensor[diagonal, diagonal] = self_mobility * identity
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

    def compute_tensor(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute the full Rotne-Prager tensor for all particle pairs (vectorised over pairs).
        Args:
            positions: Array of shape (3, n) containing particle positions
        Returns:
            Rotne-Prager tensor of shape (n, n, 3, 3)
        """
        n_particles = positions.shape[1]
        a = self._constants.a
        eta = self._constants.eta

        pos = positions.T  # (n, 3)
        r_vec = pos[:, None, :] - pos[None, :, :]  # (n, n, 3)
        r = np.linalg.norm(r_vec, axis=2)  # (n, n)
        r_safe = np.where(r > 0, r, 1.0)  # avoid div-by-zero on the diagonal (overwritten below)

        r_hat = r_vec / r_safe[:, :, None]  # (n, n, 3); zero on the diagonal
        outer = r_hat[:, :, :, None] * r_hat[:, :, None, :]  # (n, n, 3, 3)
        identity = np.identity(3)

        # Far-field form (r >= 2a).
        far_prefactor = self._prefactor / (8 * np.pi * eta * r_safe)
        far = far_prefactor[:, :, None, None] * (
            (1 + 2 * a**2 / (3 * r_safe**2))[:, :, None, None] * identity + (1 - 2 * a**2 / r_safe**2)[:, :, None, None] * outer
        )

        # Regularised overlapping form (r < 2a).
        near_prefactor = self._prefactor / (6 * np.pi * eta * a)
        near = near_prefactor * ((1 - 9 * r / (32 * a))[:, :, None, None] * identity + (3 * r / (32 * a))[:, :, None, None] * outer)

        tensor = np.where((r >= 2 * a)[:, :, None, None], far, near)

        # Overwrite the diagonal with the self-mobility tensor (the near form already yields this at r = 0).
        diagonal = np.arange(n_particles)
        tensor[diagonal, diagonal] = near_prefactor * identity
        return tensor

    @classmethod
    def create(cls, constants: GlobalConstants) -> "RotnePragerTensor":
        return cls(constants=constants, prefactor=constants.kB * constants.T)


class CachedTensor(FluidDynamicsTensorInterface):
    """
    Memoising wrapper around another tensor.
    """

    def __init__(self, tensor: FluidDynamicsTensorInterface) -> None:
        self._tensor = tensor
        self._cached_positions: np.ndarray | None = None
        self._cached_result: np.ndarray | None = None

    def compute_tensor(self, positions: np.ndarray) -> np.ndarray:
        """Return the wrapped tensor's result, reusing the previous one if positions are unchanged."""
        if self._cached_result is None or not np.array_equal(positions, self._cached_positions):
            self._cached_result = self._tensor.compute_tensor(positions)
            self._cached_positions = positions.copy()
        return self._cached_result

    @classmethod
    def create(cls, constants: GlobalConstants) -> "CachedTensor":
        raise NotImplementedError("CachedTensor wraps an existing tensor instance; construct it directly as CachedTensor(tensor).")
