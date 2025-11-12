"""Visualization components for the hydrodynamic simulation."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

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
