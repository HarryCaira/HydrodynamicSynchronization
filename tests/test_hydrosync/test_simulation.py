import numpy as np
from src.hydrosync.constants import GlobalConstants
from src.hydrosync.arrays import GridArray
from src.hydrosync.simulation import Simulation, SimulationLog

from tests.test_hydrosync.test_fluid_dynamics_tensors import MockOnesTensor
from tests.test_hydrosync.test_displacement_computation import MockDisplaceParticlesByOne


class TestSimulationLog:
    def test__add_positions(self) -> None:
        log = SimulationLog(orbit_centers=np.array([[0, 0], [0, 0]]), positions_history=[])
        positions = np.array([[1, 2], [3, 4]])
        log.add_positions(positions)
        assert len(log.positions_history) == 1
        np.testing.assert_array_equal(log.positions_history[0], positions)

    def test__synchronization_history(self) -> None:
        log = SimulationLog(orbit_centers=np.array([[0, 0], [0, 0]]), positions_history=[])
        positions1 = np.array([[1, 0], [0, 1]])
        positions2 = np.array([[0, 1], [-1, 0]])
        log.add_positions(positions1)
        log.add_positions(positions2)
        history = log.synchronization_history()
        assert len(history) == 2
        assert all(0 <= r <= 1 for r in history)


class TestSimulation:
    def test__create(self) -> None:
        constants = GlobalConstants.create(simulation_steps=10)
        mock_tensor = MockOnesTensor.create(constants)
        mock_displacements = MockDisplaceParticlesByOne.create(constants, mock_tensor, [])

        particle_array = GridArray.create(nx=2, ny=2, constants=constants)

        simulation = Simulation.create(constants, particle_array, [mock_displacements])
        assert isinstance(simulation, Simulation)

    def test__run__no_displacements(self) -> None:
        constants = GlobalConstants.create(simulation_steps=10)
        particle_array = GridArray.create(nx=2, ny=2, constants=constants)

        simulation = Simulation.create(constants, particle_array, [])

        initial_positions = particle_array.positions.copy()
        simulation.run()
        actual_positions = particle_array.positions

        assert np.array_equal(initial_positions, actual_positions)

    def test__run__displacements_are_added_once_per_step(self):
        num_steps = 10
        constants = GlobalConstants.create(simulation_steps=num_steps)
        mock_tensor = MockOnesTensor.create(constants)
        mock_displacement = MockDisplaceParticlesByOne.create(constants, mock_tensor, [])

        particle_array = GridArray.create(nx=2, ny=2, constants=constants)

        simulation = Simulation.create(constants, particle_array, [mock_displacement])

        initial_positions = particle_array.positions.copy()
        expected_positions = initial_positions + (mock_displacement.compute_displacements(initial_positions, np.zeros(0)) * num_steps)

        simulation.run()
        actual_positions = particle_array.positions

        np.testing.assert_allclose(expected_positions, actual_positions)
