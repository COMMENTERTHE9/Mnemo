"""Mnemo CLI. Subcommands: api, worker, initdb."""
import logging
import typer
import uvicorn

from mnemo.config import get_settings
from mnemo.db import init_for_settings

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def api() -> None:
    """Run the FastAPI server."""
    settings = get_settings()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "mnemo.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


@app.command()
def worker() -> None:
    """Run the background pipeline worker."""
    logging.basicConfig(level=logging.INFO)
    from mnemo.worker.runner import run_worker
    run_worker()


@app.command()
def initdb() -> None:
    """Initialize the SQLite schema (idempotent)."""
    settings = get_settings()
    conn = init_for_settings(settings)
    conn.close()
    typer.echo(f"Schema initialized at {settings.db_path}")


@app.command()
def export(video_id: str, lite: bool = False) -> None:
    """Export a video's full memory tree + raw signals to data/corpus/<video_id>.json.

    --lite drops the heavy per-landmark motion fields (pose_data,
    joint_velocities), keeping the scalar motion summary."""
    from mnemo.corpus.export import write_export
    settings = get_settings()
    conn = init_for_settings(settings)
    try:
        out_path = write_export(conn, video_id, lite=lite)
    finally:
        conn.close()
    typer.echo(f"Exported {video_id} to {out_path}")


if __name__ == "__main__":
    app()
