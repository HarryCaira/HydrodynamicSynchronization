import numpy as np
from typing import Optional
from src.arrays import ParticleArray2D, ParticleArray2DInterface
from src.constants import GlobalConstants


class TestParticleArray2D:
    def test__create__arrays_are_correct_shapes(self) -> None:
        nx = 13
        ny = 27
        constants = GlobalConstants.create()
        particle_array = ParticleArray2D.create(nx, ny, constants)

        assert particle_array.orbit_centers.shape == (3, nx * ny)
        assert particle_array.positions.shape == (3, nx * ny)

    def test__create__arrays_contain_correct_data(self) -> None:
        nx = ny = 2
        r = 0.5
        constants = GlobalConstants.create(orbit_radius=r, orbit_separation=3 * r)
        particle_array = ParticleArray2D.create(nx, ny, constants)

        expected_orbit_centers = np.array([[0.0, 0.0, 0.0], [0.0, 1.5, 0.0], [1.5, 0.0, 0.0], [1.5, 1.5, 0.0]]).T
        expected_positions = expected_orbit_centers + np.array([[r, 0.0, 0.0], [r, 0.0, 0.0], [r, 0.0, 0.0], [r, 0.0, 0.0]]).T

        np.testing.assert_array_almost_equal(particle_array.orbit_centers, expected_orbit_centers)
        np.testing.assert_array_almost_equal(particle_array.positions, expected_positions)

    def test__update_positions(self) -> None:
        array = ParticleArray2D(orbit_centers=np.zeros((3, 2)), positions=np.zeros((3, 2)))
        new_positions = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        array.update_positions(new_positions)

        np.testing.assert_array_equal(array.positions, new_positions)
        np.testing.assert_array_equal(array.orbit_centers, np.zeros((3, 2)))

    def test__positions(self) -> None:
        positions = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        array = ParticleArray2D(orbit_centers=np.zeros((3, 2)), positions=positions)

        np.testing.assert_array_equal(array.positions, positions)

    def test__orbit_centers(self) -> None:
        orbit_centers = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        array = ParticleArray2D(orbit_centers=orbit_centers, positions=np.zeros((3, 2)))

        np.testing.assert_array_equal(array.orbit_centers, orbit_centers)
