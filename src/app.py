import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from src.hydrosync.constants import GlobalConstants
from src.hydrosync.arrays import GridArray, HoneycombArray
from src.hydrosync.fluid_dynamics_tensors import OseenTensor
from src.hydrosync.force_computation import TangentialDrivingForce, RadialRestoringForce
from src.hydrosync.displacement_computation import HydrodynamicDisplacement, BrownianDisplacement
from src.hydrosync.simulation import Simulation


@st.cache_data
def cached_simulation(constants, x_particles, y_particles):
    """Cache simulation results to avoid recomputing"""
    # Setup simulation components
    particle_array = HoneycombArray.create(nx=x_particles, ny=y_particles, constants=constants, random_start_seed=1)
    fluid_tensor = OseenTensor.create(constants)

    # Create force computations
    driving_force = TangentialDrivingForce.create(constants)
    restoring_force = RadialRestoringForce.create(constants)

    # Create displacement computations
    hydro_displacement = HydrodynamicDisplacement.create(constants, fluid_tensor, [driving_force, restoring_force])
    brownian_displacement = BrownianDisplacement.create(constants, fluid_tensor)

    # Create progress container that updates less frequently
    progress_text = st.empty()
    progress_bar = st.progress(0)

    # Run simulation with reduced progress updates
    simulation = Simulation.create(
        constants=constants,
        particle_array=particle_array,
        hydrodynamic_displacements=[hydro_displacement, brownian_displacement],
        progress_bar=progress_bar,
        progress_update_freq=max(1, constants.simulation_steps // 100),  # Update ~100 times total
    )

    with st.spinner("Running simulation..."):
        log = simulation.run()

    # Clean up progress displays
    progress_text.empty()
    progress_bar.empty()

    return log


def create_animation(log):
    """Create interactive Plotly animation with both particle positions and synchronization"""
    positions_history = log.positions_history
    orbit_centers = log.orbit_centers
    n_particles = orbit_centers[0].shape[0]
    sync_values = log.synchronization_history()

    # Generate colors for particles
    rgba_colors = plt.cm.rainbow(np.linspace(0, 1, n_particles))
    colors = [f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})" for r, g, b, a in rgba_colors]

    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.15, row_heights=[0.7, 0.3], subplot_titles=("Particle Positions", "Synchronization"))

    # Create frames for both particle positions and synchronization
    frames = []
    for frame_idx, positions in enumerate(positions_history):
        frame_data = []

        # Particle position traces (in first subplot)
        for p_idx in range(n_particles):
            frame_data.append(
                dict(
                    type="scatter",
                    x=[orbit_centers[0][p_idx], positions[0][p_idx]],
                    y=[orbit_centers[1][p_idx], positions[1][p_idx]],
                    mode="lines+markers",
                    line=dict(color=colors[p_idx], width=2),
                    marker=dict(size=[8, 12], symbol=["x", "circle"], color=colors[p_idx]),
                    name=f"Particle {p_idx + 1}",
                    showlegend=(frame_idx == 0),
                    xaxis="x",
                    yaxis="y",
                )
            )

        # Synchronization trace (in second subplot)
        steps = list(range(frame_idx + 1))
        current_sync_values = sync_values[: frame_idx + 1]
        frame_data.append(
            dict(
                type="scatter", x=steps, y=current_sync_values, mode="lines", name="Synchronization", line=dict(color="rgb(31, 119, 180)", width=2), showlegend=False, xaxis="x2", yaxis="y2"
            )
        )

        frames.append(go.Frame(data=frame_data))

    # Add initial data
    for p_idx in range(n_particles):
        fig.add_trace(
            go.Scatter(
                x=[orbit_centers[0][p_idx], positions_history[0][0][p_idx]],
                y=[orbit_centers[1][p_idx], positions_history[0][1][p_idx]],
                mode="lines+markers",
                line=dict(color=colors[p_idx], width=2),
                marker=dict(size=[8, 12], symbol=["x", "circle"], color=colors[p_idx]),
                name=f"Particle {p_idx + 1}",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(go.Scatter(x=[0], y=[sync_values[0]], mode="lines", name="Synchronization", line=dict(color="rgb(31, 119, 180)", width=2), showlegend=False), row=2, col=1)

    # Set initial layout with default speed
    frame_duration = 50  # Default duration

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        updatemenus=[
            {
                "buttons": [
                    {"args": [None, {"frame": {"duration": frame_duration, "redraw": True}}], "label": "Play", "method": "animate"},
                    {"args": [[None], {"frame": {"duration": 0, "redraw": False}}], "label": "Pause", "method": "animate"},
                ],
                "type": "buttons",
                "showactive": False,
            }
        ],
    )

    # Update axes
    fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_yaxes(range=[0, 1], row=2, col=1)

    # Add frames to the figure
    fig.frames = frames

    return fig


def create_sync_plot(log):
    """Create line plot showing synchronization evolution over time"""
    sync_values = log.synchronization_history()
    steps = list(range(len(sync_values)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=sync_values, mode="lines", name="Synchronization", line=dict(color="rgb(31, 119, 180)", width=2)))

    fig.update_layout(
        title="Synchronization Evolution",
        xaxis_title="Simulation Step",
        yaxis_title="Synchronization Value",
        yaxis=dict(range=[0, 1]),  # Synchronization is between 0 and 1
        showlegend=False,
    )

    return fig


def update_animation_speed(fig, speed: float) -> None:
    """Update the animation speed of an existing figure"""
    if fig is None:
        return

    frame_duration = int(50 / speed)
    if hasattr(fig, "layout") and fig.layout.updatemenus is not None:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = frame_duration


def main() -> None:
    st.title("Simulating Hydrodynamic Coupling")

    # Sidebar for simulation parameters
    st.sidebar.header("Simulation Parameters")
    x_particles = st.sidebar.number_input("Number of particles in x-direction", min_value=1, max_value=10, value=3)
    y_particles = st.sidebar.number_input("Number of particles in y-direction", min_value=1, max_value=10, value=3)
    simulation_steps = st.sidebar.number_input("Number of simulation steps", min_value=1, max_value=10000, value=1000)

    # Animation controls in sidebar
    st.sidebar.header("Animation Controls")
    playback_speed = st.sidebar.slider("Playback Speed", min_value=0.1, max_value=5.0, value=1.0, step=0.1, help="Adjust animation speed (1.0 is normal speed, >1 is faster, <1 is slower)")

    constants = GlobalConstants.create(simulation_steps=simulation_steps)

    # Initialize session state
    if "simulation_complete" not in st.session_state:
        st.session_state.simulation_complete = False
        st.session_state.animation_fig = None
        st.session_state.log = None

    if st.sidebar.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            log = cached_simulation(constants, x_particles, y_particles)
            st.session_state.simulation_complete = True
            st.session_state.log = log
            # Create the animation figure once and store it
            st.session_state.animation_fig = create_animation(log)

    if st.session_state.simulation_complete and st.session_state.animation_fig is not None:
        # Particle animation plot
        st.subheader("Particle Positions")
        # Update the animation speed if needed
        update_animation_speed(st.session_state.animation_fig, playback_speed)
        st.plotly_chart(st.session_state.animation_fig, width="stretch")

        # Add synchronization evolution plot
        st.subheader("Synchronization Evolution")
        sync_fig = create_sync_plot(st.session_state.log)
        st.plotly_chart(sync_fig, width="stretch")

        # Add final synchronization value
        final_sync = st.session_state.log.synchronization_history()[-1]
        st.metric(label="Final Synchronization Value", value=f"{final_sync:.3f}", help="1.0 indicates perfect synchronization, 0.0 indicates no synchronization")


if __name__ == "__main__":
    main()
