from abc import ABC, abstractmethod
from typing import Optional, Any
import numpy as np
import streamlit as st
from tqdm import tqdm
from dataclasses import dataclass
import time

from src.hydrosync.constants import GlobalConstants
from src.hydrosync.arrays import ParticleArray2DInterface
from src.hydrosync.displacement_computation import ParticleDisplacementInterface


class SimulationInterface(ABC):
    @abstractmethod
    def run(self) -> None: ...

    @classmethod
    @abstractmethod
    def create(
        cls,
        constants: GlobalConstants,
        particle_array: ParticleArray2DInterface,
        displacements: ParticleDisplacementInterface,
    ) -> "SimulationInterface": ...


@dataclass
class SimulationLog:
    orbit_centers: np.ndarray
    positions_history: list[np.ndarray]

    def add_positions(self, new_positions: np.ndarray) -> None:
        self.positions_history.append(new_positions.copy())

    def synchronization_history(self) -> list[float]:
        """Compute synchronization history over time using the kuramoto condition"""
        strengths = []
        for positions in self.positions_history:
            angles = np.arctan2(
                positions[1, :] - self.orbit_centers[1, :],
                positions[0, :] - self.orbit_centers[0, :],
            )
            n_particles = len(angles)
            r = np.abs(np.sum(np.exp(1j * angles))) / n_particles
            strengths.append(r)
        return strengths


class Simulation(SimulationInterface):
    """
    Main simulation class that integrates particle positions over time
    using hydrodynamic displacements.
    """

    def __init__(
        self,
        constants: GlobalConstants,
        particle_array: ParticleArray2DInterface,
        hydrodynamic_displacements: list[ParticleDisplacementInterface],
        simulation_log: SimulationLog,
        progress_bar: Optional[Any] = None,
        progress_update_freq: int = 10,
    ):
        self._constants = constants
        self._particles = particle_array
        self._displacements = hydrodynamic_displacements
        self._log = simulation_log
        self._progress_bar = progress_bar
        self._progress_update_freq = progress_update_freq

    def run(self) -> SimulationLog:
        self._log.add_positions(self._particles.positions)
        total_steps = self._constants.simulation_steps

        total_displacements = np.zeros_like(self._particles.positions)

        for i in tqdm(range(total_steps), desc="Simulating"):
            total_displacements.fill(0)

            for displacement in self._displacements:
                total_displacements += displacement.compute_displacements(
                    self._particles.positions, self._particles.orbit_centers
                )

            new_positions = self._particles.positions + total_displacements
            self._particles.update_positions(new_positions)
            self._log.add_positions(new_positions)

            if self._progress_bar is not None and i % self._progress_update_freq == 0:
                try:
                    self._progress_bar.progress((i + 1) / total_steps)
                except Exception as e:
                    print(f"Warning: Failed to update progress bar: {str(e)}")

        if self._progress_bar is not None:
            try:
                self._progress_bar.progress(1.0)
            except Exception as e:
                print(f"Warning: Failed to update final progress: {str(e)}")

        return self._log

    @classmethod
    def create(
        cls,
        constants: GlobalConstants,
        particle_array: ParticleArray2DInterface,
        hydrodynamic_displacements: list[ParticleDisplacementInterface],
        progress_bar: Optional[Any] = None,
        progress_update_freq: int = 10,
    ) -> "Simulation":
        return cls(
            constants=constants,
            particle_array=particle_array,
            hydrodynamic_displacements=hydrodynamic_displacements,
            simulation_log=SimulationLog(particle_array.orbit_centers, []),
            progress_bar=progress_bar,
            progress_update_freq=progress_update_freq,
        )