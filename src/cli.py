import click
from src.hydrosync.constants import GlobalConstants
from src.hydrosync.arrays import GridArray, HoneycombArray, ParticleArray2DInterface
from src.hydrosync.fluid_dynamics_tensors import OseenTensor, FluidDynamicsTensorInterface
from src.hydrosync.displacement_computation import HydrodynamicDisplacement, BrownianDisplacement
from src.hydrosync.force_computation import TangentialDrivingForce, RadialRestoringForce

from src.hydrosync.simulation import Simulation


@click.command()
@click.option("-x", "--x-particles", default=3, help="Number of particles in x-direction", type=int)
@click.option("-y", "--y-particles", default=3, help="Number of particles in y-direction", type=int)
@click.option("-a", "--array-type", default="grid", help="The arrangement of particle orbit centres", type=click.Choice(["grid", "honeycomb"]))
@click.option("-t", "--tensor-type", default="oseen", help="Type of fluid dynamics tensor to use", type=click.Choice(["oseen"]))
@click.option("-s", "--simulation-steps", default=1000, help="Number of simulation steps", type=int)
@click.option("-k", "--spring-constant", default=0.001, help="Spring constant for radial restoring force", type=float)
@click.option("-f", "--driving-force", default=5.5e-9, help="Magnitude of tangential driving force in Newtons", type=float)
@click.option("-m", "--driving-force-modulation-strength", default=0.2, help="Amplitude of driving force modulation (0 to 1)", type=float)
def cli(
    x_particles: int, y_particles: int, array_type: str, tensor_type: str, simulation_steps: int, spring_constant: float, driving_force: float, driving_force_modulation_strength: float
) -> None:
    constants_kwargs = {
        "simulation_steps": simulation_steps,
        "spring_constant": spring_constant,
        "driving_force": driving_force,
        "driving_force_modulation_strength": driving_force_modulation_strength,
    }
    tensor_map: dict[str, type[FluidDynamicsTensorInterface]] = {
        "oseen": OseenTensor,
    }
    array_map: dict[str, type[ParticleArray2DInterface]] = {
        "grid": GridArray,
        "honeycomb": HoneycombArray,
    }

    constants = GlobalConstants.create(**constants_kwargs)
    tensor = tensor_map[tensor_type].create(constants)
    array = array_map[array_type].create(nx=x_particles, ny=y_particles, constants=constants, random_start_seed=1)

    simulation = Simulation.create(
        constants=constants,
        particle_array=array,
        hydrodynamic_displacements=[
            HydrodynamicDisplacement.create(
                constants=constants,
                fluid_dynamics_tensor=tensor,
                external_forces=[
                    TangentialDrivingForce.create(constants=constants),
                    RadialRestoringForce.create(constants=constants),
                ],
            ),
            BrownianDisplacement.create(constants=constants, fluid_dynamics_tensor=tensor),
        ],
    )

    click.echo("Starting simulation...")
    log = simulation.run()
    click.echo(click.style("\nSimulation completed successfully:", fg="green"))
    click.echo(f"- Particles: {x_particles}x{y_particles}")
    click.echo(f"- Simulation Steps: {simulation_steps}")
    click.echo(f"- Elapsed Time: {simulation_steps * constants.time_step:.2f} seconds")
    click.echo(f"- Final synchronization: {log.synchronization_history()[-1]:.3f}")


if __name__ == "__main__":
    cli()
