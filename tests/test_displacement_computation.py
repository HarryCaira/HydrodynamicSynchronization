import numpy as np

from src.constants import GlobalConstants

from src.displacement_computation import (
    HydrodynamicDisplacement,
    BrownianDisplacement,
    ParticleDisplacementInterface,
)
from src.fluid_dynamics_tensors import (
    FluidDynamicsTensorInterface,
    OseenTensor,
    RotnePragerTensor,
)
from src.force_computation import TangentialDrivingForce, RadialRestoringForce


class MockDisplaceParticlesByOne(ParticleDisplacementInterface):
    def compute_displacements(self, positions: np.ndarray, orbit_centers: np.ndarray) -> np.ndarray:
        return np.ones_like(positions)

    @classmethod
    def create(
        cls,
        constants: GlobalConstants,
        fluid_dynamics_tensor: FluidDynamicsTensorInterface,
        external_forces: list,
    ) -> "MockDisplaceParticlesByOne":
        return cls()


class TestHydrodynamicDisplacement:
    def test__create(self) -> None:
        constants = GlobalConstants.create()
        oseen_tensor = OseenTensor.create(constants=constants)
        hydrodynamic_displacement = HydrodynamicDisplacement.create(
            constants,
            oseen_tensor,
            external_forces=[],
        )
        assert isinstance(hydrodynamic_displacement, HydrodynamicDisplacement)

    def test__compute_displacements(self) -> None:
        constants = GlobalConstants.create()
        oseen_tensor = OseenTensor.create(constants=constants)
        hydrodynamic_displacement = HydrodynamicDisplacement.create(
            constants,
            oseen_tensor,
            external_forces=[],
        )
        positions = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        orbit_centers = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        displacements = hydrodynamic_displacement.compute_displacements(positions, orbit_centers)

        assert displacements.shape == positions.shape

    def test__compute_displacements__matches_explicit_loop(self) -> None:
        # The vectorised (einsum) drift must equal the explicit double-loop contraction
        # v_i = (dt / kT) * sum_j D_ij @ F_j.
        constants = GlobalConstants.create()
        tensor = OseenTensor.create(constants=constants)
        forces = [TangentialDrivingForce.create(constants), RadialRestoringForce.create(constants)]
        hydrodynamic_displacement = HydrodynamicDisplacement.create(constants, tensor, forces)

        rng = np.random.default_rng(0)
        positions = rng.uniform(-3e-6, 3e-6, size=(3, 5))
        orbit_centers = rng.uniform(-3e-6, 3e-6, size=(3, 5))

        result = hydrodynamic_displacement.compute_displacements(positions, orbit_centers)

        # Reference: the explicit per-pair loop the einsum replaces.
        grand_tensor = tensor.compute_tensor(positions)
        total_forces = sum(f.compute_forces(positions, orbit_centers) for f in forces)
        n = positions.shape[1]
        expected = np.zeros_like(positions)
        for i in range(n):
            for j in range(n):
                expected[:, i] += (constants.time_step * grand_tensor[i, j] @ total_forces[:, j]) / (constants.kB * constants.T)

        np.testing.assert_allclose(result, expected)


class TestBrownianDisplacement:
    def test__create(self) -> None:
        constants = GlobalConstants.create()
        oseen_tensor = OseenTensor.create(constants=constants)
        brownian_displacement = BrownianDisplacement.create(constants, oseen_tensor, [])
        assert isinstance(brownian_displacement, BrownianDisplacement)

    def test__compute_displacements(self) -> None:
        constants = GlobalConstants.create()
        oseen_tensor = OseenTensor.create(constants=constants)
        brownian_displacement = BrownianDisplacement.create(constants, oseen_tensor, [])

        positions = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        displacements = brownian_displacement.compute_displacements(positions, positions)

        assert displacements.shape == positions.shape

    def test__compute_displacements__has_expected_covariance(self) -> None:
        # The Brownian step must sample N(0, 2 dt D), so the empirical covariance of many
        # draws should match the 3n x 3n diffusion-tensor covariance. Rotne-Prager is used
        # because its covariance is positive-definite (no eigenvalue clipping in the sampler).
        np.random.seed(0)
        constants = GlobalConstants.create()
        tensor = RotnePragerTensor.create(constants=constants)
        brownian = BrownianDisplacement.create(constants, tensor, [])

        positions = np.array([[0.0, 5e-6], [0.0, 0.0], [0.0, 0.0]])  # 2 particles, separation > 2a
        n = positions.shape[1]
        expected_cov = 2 * constants.time_step * tensor.compute_tensor(positions).transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)

        samples = np.array([brownian.compute_displacements(positions, positions).T.reshape(3 * n) for _ in range(30000)])
        empirical_cov = np.cov(samples, rowvar=False)

        np.testing.assert_allclose(empirical_cov, expected_cov, atol=0.1 * expected_cov.max())
