import numpy as np
import pytest

from src.fluid_dynamics_tensors import OseenTensor, RotnePragerTensor, CachedTensor
from src.constants import GlobalConstants
from src.fluid_dynamics_tensors import FluidDynamicsTensorInterface


class MockOnesTensor(FluidDynamicsTensorInterface):
    def compute_tensor(self, positions: np.ndarray) -> np.ndarray:
        n_particles = positions.shape[1]
        tensor = np.ones((n_particles, n_particles, 3, 3))
        return tensor

    @classmethod
    def create(cls, constants: GlobalConstants) -> "MockOnesTensor":
        return cls()


class CountingTensor(FluidDynamicsTensorInterface):
    """Counts compute_tensor calls and returns a distinct value each call,
    so a stale cache (returning an old value) is detectable."""

    def __init__(self) -> None:
        self.calls = 0

    def compute_tensor(self, positions: np.ndarray) -> np.ndarray:
        self.calls += 1
        n_particles = positions.shape[1]
        return np.full((n_particles, n_particles, 3, 3), float(self.calls))

    @classmethod
    def create(cls, constants: GlobalConstants) -> "CountingTensor":
        return cls()


class TestOseenTensor:
    def test__create(self) -> None:
        constants = GlobalConstants.create()
        tensor = OseenTensor.create(constants)
        assert isinstance(tensor, OseenTensor)

    def test__compute_tensor__outputs_expected_shape(self) -> None:
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]).T
        tensor = OseenTensor.create(GlobalConstants.create())
        result = tensor.compute_tensor(positions)

        expected_shape = (positions.shape[1], positions.shape[1], 3, 3)
        assert result.shape == expected_shape

    def test__compute_tensor__correct_values(self) -> None:
        constants = GlobalConstants.create(eta=1.0)
        tensor = OseenTensor.create(constants)

        particle_1 = np.array([0.0, 0.0, 0.0])
        particle_2 = np.array([1.0, 0.0, 0.0])
        positions = np.array([particle_1, particle_2]).T
        result = tensor.compute_tensor(positions)

        expected_self_mobility_tensor = (constants.kB * constants.T / (6 * np.pi * constants.eta * constants.a)) * np.identity(3)
        np.testing.assert_array_almost_equal(result[0, 0], expected_self_mobility_tensor)
        np.testing.assert_array_almost_equal(result[1, 1], expected_self_mobility_tensor)

        r = np.linalg.norm(particle_2 - particle_1)
        expected_cross_mobility_tensor = (constants.kB * constants.T / (8 * np.pi * constants.eta * r)) * (
            np.identity(3) + np.outer((particle_2 - particle_1) / r, (particle_2 - particle_1) / r)
        )
        np.testing.assert_array_almost_equal(result[0, 1], expected_cross_mobility_tensor)
        np.testing.assert_array_almost_equal(result[1, 0], expected_cross_mobility_tensor)


class TestRotnePragerTensor:
    def test__create(self) -> None:
        constants = GlobalConstants.create()
        tensor = RotnePragerTensor.create(constants)
        assert isinstance(tensor, RotnePragerTensor)

    def test__compute_tensor__outputs_expected_shape(self) -> None:
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]).T
        tensor = RotnePragerTensor.create(GlobalConstants.create())
        result = tensor.compute_tensor(positions)

        expected_shape = (positions.shape[1], positions.shape[1], 3, 3)
        assert result.shape == expected_shape

    def test__compute_tensor__self_mobility_matches_stokes_einstein(self) -> None:
        constants = GlobalConstants.create(eta=1.0)
        tensor = RotnePragerTensor.create(constants)

        positions = np.array([[0.0, 0.0, 0.0], [10e-6, 0.0, 0.0]]).T
        result = tensor.compute_tensor(positions)

        expected_self_mobility_tensor = (constants.kB * constants.T / (6 * np.pi * constants.eta * constants.a)) * np.identity(3)
        np.testing.assert_array_almost_equal(result[0, 0], expected_self_mobility_tensor)
        np.testing.assert_array_almost_equal(result[1, 1], expected_self_mobility_tensor)

    def test__compute_tensor__non_overlapping_cross_mobility(self) -> None:
        # Separation well beyond 2a so the far-field (non-overlapping) branch is used.
        constants = GlobalConstants.create(eta=1.0, a=1e-6)
        tensor = RotnePragerTensor.create(constants)
        a = constants.a

        particle_1 = np.array([0.0, 0.0, 0.0])
        particle_2 = np.array([10e-6, 0.0, 0.0])
        positions = np.array([particle_1, particle_2]).T
        result = tensor.compute_tensor(positions)

        r_ij = particle_1 - particle_2
        r = np.linalg.norm(r_ij)
        r_hat = r_ij / r
        expected = (constants.kB * constants.T / (8 * np.pi * constants.eta * r)) * (
            (1 + 2 * a**2 / (3 * r**2)) * np.identity(3) + (1 - 2 * a**2 / r**2) * np.outer(r_hat, r_hat)
        )
        np.testing.assert_array_almost_equal(result[0, 1], expected)
        np.testing.assert_array_almost_equal(result[1, 0], expected)

    def test__compute_tensor__overlapping_cross_mobility(self) -> None:
        # Separation below 2a so the regularised (overlapping) branch is used.
        constants = GlobalConstants.create(eta=1.0, a=1e-6)
        tensor = RotnePragerTensor.create(constants)
        a = constants.a

        particle_1 = np.array([0.0, 0.0, 0.0])
        particle_2 = np.array([a, 0.0, 0.0])  # r = a < 2a
        positions = np.array([particle_1, particle_2]).T
        result = tensor.compute_tensor(positions)

        r_ij = particle_1 - particle_2
        r = np.linalg.norm(r_ij)
        r_hat = r_ij / r
        expected = (constants.kB * constants.T / (6 * np.pi * constants.eta * a)) * (
            (1 - 9 * r / (32 * a)) * np.identity(3) + (3 * r / (32 * a)) * np.outer(r_hat, r_hat)
        )
        np.testing.assert_array_almost_equal(result[0, 1], expected)
        np.testing.assert_array_almost_equal(result[1, 0], expected)

    def test__compute_tensor__branches_continuous_at_contact(self) -> None:
        # The two branches must agree at r = 2a (particle contact).
        constants = GlobalConstants.create(eta=1.0, a=1e-6)
        tensor = RotnePragerTensor.create(constants)
        a = constants.a

        eps = 1e-12  # tiny offset to probe either side of the r = 2a boundary
        just_below = tensor._cross_mobility(np.array([2 * a - eps, 0.0, 0.0]))
        just_above = tensor._cross_mobility(np.array([2 * a + eps, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(just_below, just_above)

    def test__compute_tensor__is_symmetric_and_positive_definite(self) -> None:
        # The 3n x 3n grand mobility matrix should be symmetric positive-definite,
        # which is the defining advantage of Rotne-Prager over the Oseen tensor.
        constants = GlobalConstants.create()
        tensor = RotnePragerTensor.create(constants)

        positions = np.array([[0.0, 0.0, 0.0], [3e-6, 0.0, 0.0], [0.0, 3e-6, 0.0]]).T
        n = positions.shape[1]
        result = tensor.compute_tensor(positions)

        grand = np.block([[result[i, j] for j in range(n)] for i in range(n)])
        np.testing.assert_array_almost_equal(grand, grand.T)
        eigenvalues = np.linalg.eigvalsh(grand)
        assert np.all(eigenvalues > 0)


class TestCachedTensor:
    def test__delegates_to_wrapped_tensor(self) -> None:
        positions = np.array([[0.0, 0.0, 0.0], [3e-6, 0.0, 0.0]]).T
        inner = OseenTensor.create(GlobalConstants.create())
        cached = CachedTensor(inner)

        np.testing.assert_array_equal(cached.compute_tensor(positions), inner.compute_tensor(positions))

    def test__same_positions_computes_once(self) -> None:
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).T
        inner = CountingTensor()
        cached = CachedTensor(inner)

        first = cached.compute_tensor(positions)
        second = cached.compute_tensor(positions)  # identical positions -> cache hit

        assert inner.calls == 1
        np.testing.assert_array_equal(first, second)  # cached value, not recomputed

    def test__copied_positions_still_hit_cache(self) -> None:
        # Cache validity is by value, so an equal-but-distinct array still hits.
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).T
        inner = CountingTensor()
        cached = CachedTensor(inner)

        cached.compute_tensor(positions)
        cached.compute_tensor(positions.copy())

        assert inner.calls == 1

    def test__changed_positions_recompute(self) -> None:
        inner = CountingTensor()
        cached = CachedTensor(inner)

        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).T
        cached.compute_tensor(positions)
        moved = positions + 1.0  # new positions -> cache miss
        result = cached.compute_tensor(moved)

        assert inner.calls == 2
        np.testing.assert_array_equal(result, np.full_like(result, 2.0))  # fresh value

    def test__create_is_not_supported(self) -> None:
        with pytest.raises(NotImplementedError):
            CachedTensor.create(GlobalConstants.create())
