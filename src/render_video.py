"""Render a saved simulation log (.npz) to a video.

Usage:
    uv run python -m src.render_video LOG.npz -o run.mp4
"""

import click

from src.simulation import SimulationLog
from src.visualization import SimulationVideo


@click.command()
@click.argument("log_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", default="simulation.mp4", help="Output video path (.mp4 via ffmpeg, or .gif)", type=click.Path(dir_okay=False, writable=True))
@click.option("--fps", default=20, help="Frames per second", type=int)
@click.option("--max-frames", default=300, help="Cap on rendered frames; longer runs are sub-sampled", type=int)
@click.option("--grid-resolution", default=120, help="Resolution of the local-order background field", type=int)
def render(log_path: str, output: str, fps: int, max_frames: int, grid_resolution: int) -> None:
    log = SimulationLog.load(log_path)
    video = SimulationVideo.from_log(log)

    click.echo(f"Rendering {len(log.positions_history)} frames from {log_path}...")
    video.render(output, fps=fps, max_frames=max_frames, grid_resolution=grid_resolution)
    click.echo(click.style(f"Saved video to: {output}", fg="green"))


if __name__ == "__main__":
    render()
