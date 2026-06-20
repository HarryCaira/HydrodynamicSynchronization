import numpy as np

from src.visualization import SimulationVideo
from src.simulation import SimulationLog


def _toy_log(n_frames: int = 5, n: int = 4) -> SimulationLog:
    """A small run: particles on a grid, each rotating around its centre."""
    centres = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])[:, :n]
    history = []
    for frame in range(n_frames):
        angle = 0.3 * frame
        offset = np.zeros_like(centres)
        offset[0] = 0.2 * np.cos(angle)
        offset[1] = 0.2 * np.sin(angle)
        history.append(centres + offset)
    return SimulationLog(orbit_centers=centres, positions_history=history)


class TestSimulationVideo:
    def test__local_order_field_is_bounded(self) -> None:
        # The local order parameter is a coherence in [0, 1]; density is normalised to <= 1.
        log = _toy_log()
        video = SimulationVideo.from_log(log)
        grid_x, grid_y = np.meshgrid(np.linspace(-1, 2, 20), np.linspace(-1, 2, 20))

        order, density = video._local_order_field(log.positions_history[0], grid_x, grid_y, sigma=1.0)

        assert order.shape == grid_x.shape
        assert np.all(order >= -1e-9) and np.all(order <= 1 + 1e-9)
        assert np.all(density >= 0) and np.isclose(density.max(), 1.0)

    def test__identical_phases_give_full_local_order(self) -> None:
        # If every particle shares the same orbital phase, local coherence is 1 everywhere.
        log = _toy_log()
        # Place all particles at the same offset (+x) from their centres -> identical phase.
        log.positions_history = [log.orbit_centers + np.array([[0.2], [0.0], [0.0]])]
        video = SimulationVideo.from_log(log)
        grid_x, grid_y = np.meshgrid(np.linspace(-1, 2, 15), np.linspace(-1, 2, 15))

        order, _ = video._local_order_field(log.positions_history[0], grid_x, grid_y, sigma=1.0)
        np.testing.assert_allclose(order, 1.0, atol=1e-9)

    def test__render_writes_a_video_file(self, tmp_path) -> None:
        # Render to .gif (pillow writer is always available, unlike ffmpeg).
        log = _toy_log(n_frames=4)
        video = SimulationVideo.from_log(log)
        output = tmp_path / "run.gif"

        video.render(str(output), fps=4, grid_resolution=24)

        assert output.exists()
        assert output.stat().st_size > 0

    def test__frame_indices_subsamples_long_runs(self) -> None:
        log = _toy_log(n_frames=1000)
        video = SimulationVideo.from_log(log)
        assert len(video._frame_indices(max_frames=100)) <= 100
        assert video._frame_indices(max_frames=10_000) == list(range(1000))
