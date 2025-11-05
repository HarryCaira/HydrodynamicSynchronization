from dataclasses import dataclass


@dataclass
class GlobalConstants:
    """Physical constants and simulation parameters"""

    kB: float  # Boltzmann constant in J/K
    T: float  # Temperature in K
    eta: float  # Dynamic viscosity in Pa·s
    spring_constant: float  # hooke's law spring constant in N/m
    a: float

    time_step: float  # Time step in seconds
    simulation_steps: int  # Number of simulation steps

    orbit_radius: float  # Orbit radius in meters
    orbit_separation: float  # Distance between orbit centers in meters

    driving_force: float  # Magnitude of the tangential driving force in Newtons
    driving_force_modulation_strength: float

    @classmethod
    def create(
        cls,
        kB: float = 1.380649e-23,
        T: float = 298.15,
        eta: float = 85e-3,
        spring_constant: float = 0.001,
        a: float = 2e-6,
        time_step: float = 1e-5,
        simulation_steps: int = 10_000,
        orbit_radius: float = 2e-6,
        orbit_separation: float = 6e-6,
        driving_force: float = 5.5e-9,
        driving_force_modulation_strength: float = 0.2,
    ) -> "GlobalConstants":
        return cls(
            kB=kB,
            T=T,
            eta=eta,
            spring_constant=spring_constant,
            a=a,
            time_step=time_step,
            simulation_steps=simulation_steps,
            orbit_radius=orbit_radius,
            orbit_separation=orbit_separation,
            driving_force=driving_force,
            driving_force_modulation_strength=driving_force_modulation_strength,
        )
