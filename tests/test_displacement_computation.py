import numpy as np

from src.constants import GlobalConstants

from src.displacement_computation import HydrodynamicDisplacement, BrownianDisplacement, ParticleDisplacementInterface
from src.fluid_dynamics_tensors import FluidDynamicsTensorInterface, OseenTensor


class MockDisplaceParticlesByOne(ParticleDisplacementInterface):
    def compute_displacements(self, positions: np.ndarray, orbit_centers: np.ndarray) -> np.ndarray:
        return np.ones_like(positions)

    @classmethod
    def create(cls, constants: GlobalConstants, fluid_dynamics_tensor: FluidDynamicsTensorInterface, external_forces: list) -> "MockDisplaceParticlesByOne":
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
        # TODO: Further assertions to be made based on expected statistical properties of Brownian displacements
