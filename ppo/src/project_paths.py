"""Locations of bundled inputs and default outputs, independent of the caller's cwd."""

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PPO_DIR = SRC_DIR.parent


def project_path(relative_path):
    """Return a string path under ppo/; explicit CLI paths remain caller-relative."""
    return str(PPO_DIR / relative_path)
