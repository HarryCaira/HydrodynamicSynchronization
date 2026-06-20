"""Visualization components for the hydrodynamic simulation."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.simulation import SimulationLog


@dataclass
class PlotColors:
    """Color generation for particle visualization."""

    n_particles: int
    _colors: Optional[List[str]] = None

    def __post_init__(self) -> None:
        rgba_colors = plt.cm.rainbow(np.linspace(0, 1, self.n_particles))
        self._colors = [f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})" for r, g, b, _ in rgba_colors]

    def get_color(self, index: int) -> str:
        """Get color for a specific particle index."""
        if not self._colors:
            raise RuntimeError("Colors not initialized")
        return self._colors[index % self.n_particles]


class AnimationBuilder:
    """Builds Plotly animations for particle visualization."""

    def __init__(
        self,
        positions_history: List[np.ndarray],
        orbit_centers: np.ndarray,
        sync_values: np.ndarray,
    ) -> None:
        """Initialize the animation builder."""
        self.positions_history = positions_history
        self.orbit_centers = orbit_centers
        self.sync_values = sync_values
        self.n_particles = orbit_centers[0].shape[0]
        self.colors = PlotColors(self.n_particles)
        self.fig = make_subplots(
            rows=2,
            cols=1,
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3],
            subplot_titles=("Particle Positions", "Synchronization"),
        )

    def _create_particle_trace(self, p_idx: int, positions: np.ndarray, frame_idx: int = 0) -> Dict[str, Any]:
        """Create a scatter trace for a single particle."""
        return {
            "type": "scatter",
            "x": [self.orbit_centers[0][p_idx], positions[0][p_idx]],
            "y": [self.orbit_centers[1][p_idx], positions[1][p_idx]],
            "mode": "lines+markers",
            "line": dict(color=self.colors.get_color(p_idx), width=2),
            "marker": dict(
                size=[8, 12],
                symbol=["x", "circle"],
                color=self.colors.get_color(p_idx),
            ),
            "name": f"Particle {p_idx + 1}",
            "showlegend": (frame_idx == 0),
            "xaxis": "x",
            "yaxis": "y",
        }

    def _create_sync_trace(self, frame_idx: int) -> Dict[str, Any]:
        """Create a synchronization trace."""
        steps = list(range(frame_idx + 1))
        current_sync_values = self.sync_values[: frame_idx + 1]
        return {
            "type": "scatter",
            "x": steps,
            "y": current_sync_values,
            "mode": "lines",
            "name": "Synchronization",
            "line": dict(color="rgb(31, 119, 180)", width=2),
            "showlegend": False,
            "xaxis": "x2",
            "yaxis": "y2",
        }

    def _create_frames(self) -> List[go.Frame]:
        """Create animation frames."""
        frames = []
        for frame_idx, positions in enumerate(self.positions_history):
            frame_data = [self._create_particle_trace(p_idx, positions, frame_idx) for p_idx in range(self.n_particles)]
            frame_data.append(self._create_sync_trace(frame_idx))
            frames.append(go.Frame(data=frame_data))
        return frames

    def _add_initial_traces(self) -> None:
        """Add initial traces to the figure."""
        for p_idx in range(self.n_particles):
            self.fig.add_trace(
                go.Scatter(
                    x=[
                        self.orbit_centers[0][p_idx],
                        self.positions_history[0][0][p_idx],
                    ],
                    y=[
                        self.orbit_centers[1][p_idx],
                        self.positions_history[0][1][p_idx],
                    ],
                    mode="lines+markers",
                    line=dict(color=self.colors.get_color(p_idx), width=2),
                    marker=dict(
                        size=[8, 12],
                        symbol=["x", "circle"],
                        color=self.colors.get_color(p_idx),
                    ),
                    name=f"Particle {p_idx + 1}",
                ),
                row=1,
                col=1,
            )

        self.fig.add_trace(
            go.Scatter(
                x=[0],
                y=[self.sync_values[0]],
                mode="lines",
                name="Synchronization",
                line=dict(color="rgb(31, 119, 180)", width=2),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    def build(self, default_duration: int = 50) -> go.Figure:
        """Build the complete animation figure."""
        frames = self._create_frames()
        self._add_initial_traces()

        self.fig.update_layout(
            height=800,
            showlegend=True,
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {
                                        "duration": default_duration,
                                        "redraw": True,
                                    }
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {"frame": {"duration": 0, "redraw": False}},
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "type": "buttons",
                    "showactive": False,
                }
            ],
        )

        self.fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
        self.fig.update_yaxes(range=[0, 1], row=2, col=1)
        self.fig.frames = frames

        return self.fig


class SynchronizationPlot:
    """Creates synchronization evolution plots."""

    @staticmethod
    def create(sync_values: np.ndarray) -> go.Figure:
        """Create a synchronization evolution plot."""
        steps = list(range(len(sync_values)))
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=sync_values,
                mode="lines",
                name="Synchronization",
                line=dict(color="rgb(31, 119, 180)", width=2),
            )
        )

        fig.update_layout(
            title="Synchronization Evolution",
            xaxis_title="Simulation Step",
            yaxis_title="Synchronization Value",
            yaxis=dict(range=[0, 1]),
            showlegend=False,
        )

        return fig


class SimulationVideo:
    """Renders a simulation run to a video file (matplotlib animation).

    Each frame shows, in the orbit plane:
      - the orbit radius of every particle as a vector from its centre to its
        current position, coloured by orbital phase (so equal colours are in phase);
      - a translucent background field of the *local* Kuramoto order parameter, a
        spatially-weighted phase coherence in [0, 1] that is bright where neighbouring
        oscillators are locally synchronised, faded where there are no particles;
      - a companion panel tracking the global synchronization over time.

    The local order field at a point p is the phase coherence of the particles,
    weighted by a Gaussian in their distance to p:
        r_local(p) = | sum_i w_i(p) e^{i theta_i} | / sum_i w_i(p),
    with w_i(p) = exp(-|p - x_i|^2 / 2 sigma^2). This is the standard global Kuramoto
    order parameter restricted to a neighbourhood, so r_local -> 1 where nearby
    oscillators share a phase and -> 0 where they are scrambled.
    """

    PHASE_CMAP = "twilight"  # cyclic: equal colour == equal phase
    FIELD_CMAP = "viridis"  # sequential: bright == locally synchronised

    def __init__(self, positions_history: List[np.ndarray], orbit_centers: np.ndarray, sync_values: np.ndarray) -> None:
        self.positions_history = positions_history
        self.orbit_centers = orbit_centers
        self.sync_values = sync_values
        self.n_particles = orbit_centers.shape[1]

    @classmethod
    def from_log(cls, log: SimulationLog) -> "SimulationVideo":
        return cls(log.positions_history, log.orbit_centers, np.array(log.synchronization_history()))

    def _orbital_phases(self, positions: np.ndarray) -> np.ndarray:
        """Phase of each particle: the angle of its radius vector about its centre."""
        return np.arctan2(positions[1] - self.orbit_centers[1], positions[0] - self.orbit_centers[0])

    def _kernel_bandwidth(self) -> float:
        """Gaussian bandwidth sigma: just under the nearest-neighbour centre spacing."""
        centres = self.orbit_centers[:2].T  # (n, 2)
        if self.n_particles < 2:
            return float(np.max(np.abs(self.positions_history[0][:2] - self.orbit_centers[:2])) + 1e-12)
        pairwise = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
        np.fill_diagonal(pairwise, np.inf)
        return 0.9 * float(pairwise.min())

    def _local_order_field(self, positions: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (local order parameter, normalised particle density) over the grid."""
        phasors = np.exp(1j * self._orbital_phases(positions))  # (n,)
        dx = grid_x[..., None] - positions[0][None, None, :]  # (H, W, n)
        dy = grid_y[..., None] - positions[1][None, None, :]
        weights = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))  # (H, W, n)
        weight_sum = weights.sum(axis=2)
        order = np.abs((weights * phasors[None, None, :]).sum(axis=2)) / np.where(weight_sum > 0, weight_sum, 1.0)
        density = weight_sum / weight_sum.max()  # fade regions with no particles
        return order, density

    def _frame_indices(self, max_frames: int) -> List[int]:
        """Sub-sample frames so long runs still render to a reasonably short video."""
        total = len(self.positions_history)
        if total <= max_frames:
            return list(range(total))
        stride = int(np.ceil(total / max_frames))
        return list(range(0, total, stride))

    def render(self, output_path: str, fps: int = 20, max_frames: int = 300, grid_resolution: int = 120) -> None:
        """Render the run to a video at output_path (.mp4 via ffmpeg, or .gif)."""
        frame_indices = self._frame_indices(max_frames)
        sigma = self._kernel_bandwidth()

        # Bounding box covering every centre and position, with a margin.
        all_positions = np.concatenate(self.positions_history, axis=1)  # (3, total*n)
        xs = np.concatenate([self.orbit_centers[0], all_positions[0]])
        ys = np.concatenate([self.orbit_centers[1], all_positions[1]])
        margin = 0.6 * sigma
        x_min, x_max = xs.min() - margin, xs.max() + margin
        y_min, y_max = ys.min() - margin, ys.max() + margin
        grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, grid_resolution), np.linspace(y_min, y_max, grid_resolution))

        fig, (ax, sync_ax) = plt.subplots(2, 1, figsize=(8, 9), gridspec_kw={"height_ratios": [4, 1]})

        # Spatial panel: local-order field + phase-coloured radius vectors.
        order0, density0 = self._local_order_field(self.positions_history[frame_indices[0]], grid_x, grid_y, sigma)
        field_image = ax.imshow(
            order0, extent=(x_min, x_max, y_min, y_max), origin="lower", cmap=self.FIELD_CMAP, vmin=0.0, vmax=1.0, alpha=np.clip(density0, 0, 1), aspect="equal", zorder=0
        )
        ax.scatter(self.orbit_centers[0], self.orbit_centers[1], c="black", marker="x", s=30, linewidths=1.2, zorder=2)
        phases0 = self._orbital_phases(self.positions_history[frame_indices[0]])
        quiver = ax.quiver(
            self.orbit_centers[0],
            self.orbit_centers[1],
            self.positions_history[frame_indices[0]][0] - self.orbit_centers[0],
            self.positions_history[frame_indices[0]][1] - self.orbit_centers[1],
            phases0,
            cmap=self.PHASE_CMAP,
            clim=(-np.pi, np.pi),
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.006,
            zorder=3,
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.colorbar(field_image, ax=ax, fraction=0.046, pad=0.02, label="local order parameter r")
        fig.colorbar(quiver, ax=ax, fraction=0.046, pad=0.08, label="orbital phase (rad)")

        # Sync panel: global synchronization over time with a progress marker.
        steps = np.arange(len(self.sync_values))
        sync_ax.plot(steps, self.sync_values, color="tab:blue", lw=1.5)
        sync_ax.set_xlim(0, max(1, len(self.sync_values) - 1))
        sync_ax.set_ylim(0, 1)
        sync_ax.set_xlabel("step")
        sync_ax.set_ylabel("global r")
        progress_line = sync_ax.axvline(frame_indices[0], color="crimson", lw=1.5)

        def update(frame_index: int):
            positions = self.positions_history[frame_index]
            order, density = self._local_order_field(positions, grid_x, grid_y, sigma)
            field_image.set_data(order)
            field_image.set_alpha(np.clip(density, 0, 1))
            quiver.set_UVC(positions[0] - self.orbit_centers[0], positions[1] - self.orbit_centers[1], self._orbital_phases(positions))
            progress_line.set_xdata([frame_index, frame_index])
            ax.set_title(f"step {frame_index}/{len(self.positions_history) - 1}    global r = {self.sync_values[frame_index]:.3f}")
            return field_image, quiver, progress_line

        animation = FuncAnimation(fig, update, frames=frame_indices, blit=False)
        writer = "ffmpeg" if output_path.endswith(".mp4") else "pillow"
        animation.save(output_path, writer=writer, fps=fps, dpi=110)
        plt.close(fig)
