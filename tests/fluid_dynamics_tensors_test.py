from src.fluid_dynamics_tensors import OseenTensor
from src.constants import GlobalConstants
import numpy as np


class TestOseenTensor:
    def test__create(self) -> None:
        constants = GlobalConstants()
        tensor = OseenTensor.create(constants)
        assert isinstance(tensor, OseenTensor)

    def test__compute_tensor__outputs_expected_shape(self) -> None:
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]).T
        tensor = OseenTensor.create(GlobalConstants())
        result = tensor.compute_tensor(positions)

        expected_shape = (positions.shape[1], positions.shape[1], 3, 3)
        assert result.shape == expected_shape

    def test__compute_tensor__correct_values(self) -> None:
        constants = GlobalConstants(eta=1.0)
        tensor = OseenTensor.create(constants)

        particle_1 = np.array([0.0, 0.0, 0.0])
        particle_2 = np.array([1.0, 0.0, 0.0])
        positions = np.array([particle_1, particle_2]).T
        result = tensor.compute_tensor(positions)

        expected_self_mobility_tensor = (
            constants.kB * constants.T / (6 * np.pi * constants.eta * constants.a)
        ) * np.identity(3)
        np.testing.assert_array_almost_equal(result[0, 0], expected_self_mobility_tensor)
        np.testing.assert_array_almost_equal(result[1, 1], expected_self_mobility_tensor)

        r = np.linalg.norm(particle_2 - particle_1)
        expected_cross_mobility_tensor = (
            constants.kB * constants.T / (8 * np.pi * constants.eta * r)
        ) * (
            np.identity(3) + np.outer((particle_2 - particle_1) / r, (particle_2 - particle_1) / r)
        )
        np.testing.assert_array_almost_equal(result[0, 1], expected_cross_mobility_tensor)
        np.testing.assert_array_almost_equal(result[1, 0], expected_cross_mobility_tensor)
