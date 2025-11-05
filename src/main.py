from src.constants import GlobalConstants
from src.arrays import ParticleArray2D
from src.fluid_dynamics_tensors import OseenTensor
from src.force_computation import TangentialDrivingForce, RadialRestoringForce
from src.displacement_computation import HydrodynamicDisplacement, BrownianDisplacement
from src.simulation import Simulation


def main() -> None:
    constants = GlobalConstants.create()
    oseen_tensor = OseenTensor.create(constants)
    simulation = Simulation.create(
        constants=constants,
        particle_array=ParticleArray2D.create(nx=3, ny=3, constants=constants, random_seed=42),
        hydrodynamic_displacements=[
            HydrodynamicDisplacement.create(
                constants=constants,
                fluid_dynamics_tensor=oseen_tensor,
                external_forces=[
                    TangentialDrivingForce.create(constants=constants),
                    RadialRestoringForce.create(constants=constants),
                ],
            ),
            BrownianDisplacement.create(
                constants=constants,
                fluid_dynamics_tensor=oseen_tensor,
            ),
        ],
    )
    log = simulation.run()
    # TODO: visualise log results


if __name__ == "__main__":
    main()
