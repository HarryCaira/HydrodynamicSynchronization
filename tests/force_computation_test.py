import numpy as np

from src.constants import GlobalConstants
from src.force_computation import TangentialDrivingForce, RadialRestoringForce


class TestTangentialDrivingForce:
    def test__create(self) -> None:
        constants = GlobalConstants()
        driving_force = TangentialDrivingForce.create(constants)
        assert isinstance(driving_force, TangentialDrivingForce)

    def test__compute_forces__assert_force_is_tangential(self) -> None:
        force_strength = 1e-8
        driving_force = TangentialDrivingForce(force_strength=force_strength, modulation_weight=0.2)

        positions = np.array([[1.0, 0.0, 0.0]]).T
        orbit_centers = np.array([[0.0, 0.0, 0.0]]).T
        forces = driving_force.compute_forces(positions, orbit_centers)

        expected_forces = np.array([[0.0, force_strength, 0.0]]).T
        np.testing.assert_array_almost_equal(forces, expected_forces)

    def test__compute_forces__assert_force_is_zero_with_zero_strength(self) -> None:
        driving_force = TangentialDrivingForce(force_strength=0, modulation_weight=0.2)

        positions = np.array([[1.0, 0.0, 0.0]]).T
        orbit_centers = np.array([[0.0, 0.0, 0.0]]).T
        forces = driving_force.compute_forces(positions, orbit_centers)

        expected_forces = np.array([[0.0, 0.0, 0.0]]).T
        np.testing.assert_array_almost_equal(forces, expected_forces)


class TestRadialRestoringForce:
    def test__create(self) -> None:
        constants = GlobalConstants()
        restoring_force = RadialRestoringForce(constants, radius=1.0)
        assert isinstance(restoring_force, RadialRestoringForce)

    def test__compute_forces__assert_force_is_zero_with_no_extension(self) -> None:
        spring_constant = 0.1
        radius = 1.0
        force = RadialRestoringForce(spring_constant, radius)

        positions = np.array([[1.0, 0.0, 0.0]]).T
        orbit_centers = np.array([[0.0, 0.0, 0.0]]).T
        forces = force.compute_forces(positions, orbit_centers)

        expected_forces = np.array([[0.0, 0.0, 0.0]]).T
        np.testing.assert_array_almost_equal(forces, expected_forces)

    def test__compute_forces__assert_force_is_negative_with_positive_extension(
        self,
    ) -> None:
        spring_constant = 0.1
        radius = 1.0
        force = RadialRestoringForce(spring_constant, radius)

        positions = np.array([[1.1, 0.0, 0.0]]).T
        orbit_centers = np.array([[0.0, 0.0, 0.0]]).T
        forces = force.compute_forces(positions, orbit_centers)

        expected_forces = np.array([[-0.01, 0.0, 0.0]]).T
        np.testing.assert_array_almost_equal(forces, expected_forces)

    def test__compute_forces__assert_force_is_positive_with_negative_extension(
        self,
    ) -> None:
        spring_constant = 0.1
        radius = 1.0
        force = RadialRestoringForce(spring_constant, radius)

        positions = np.array([[0.9, 0.0, 0.0]]).T
        orbit_centers = np.array([[0.0, 0.0, 0.0]]).T
        forces = force.compute_forces(positions, orbit_centers)

        expected_forces = np.array([[0.01, 0.0, 0.0]]).T
        np.testing.assert_array_almost_equal(forces, expected_forces)
