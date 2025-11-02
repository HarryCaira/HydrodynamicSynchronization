from dataclasses import dataclass


@dataclass
class GlobalConstants:
    """Physical constants and simulation parameters"""

    kB: float = 1.380649e-23  # Boltzmann constant in J/K
    T: float = 298.15  # Temperature in K
    eta: float = 85e-3  # Dynamic viscosity in Pa·s
    spring_constant: float = 0.001  # hooke's law spring constant in N/m
    a: float = 2e-6

    time_step: float = 1e-5  # Time step in seconds
    simulation_steps = 10_000  # Number of simulation steps

    orbit_radius: float = 2e-6  # Orbit radius in meters
    orbit_separation: float = 3 * orbit_radius  # Distance between orbit centers in meters

    driving_force: float = 5.5e-9  # Magnitude of the tangential driving force in Newtons
    driving_force_modulation_strength: float = 0.2
