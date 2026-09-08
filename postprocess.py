import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dolfin import (
    File,
    FunctionSpace,
    Function,
    project,
    plot,
    sqrt,
)

from config import Config


def save_phase_results(
    phase: str,
    w: Function,
    mesh,
    cfg: Config,
    metrics: dict
):
    """
    Сохраняет:
    - velocity .pvd;
    - pressure .pvd;
    - PNG со скоростью.
    """

    u, p = w.split()

    velocity_path = os.path.join(cfg.output_dir, f"velocity_{phase}.pvd")
    pressure_path = os.path.join(cfg.output_dir, f"pressure_{phase}.pvd")
    png_path = os.path.join(cfg.output_dir, f"speed_{phase}.png")

    File(velocity_path) << u
    File(pressure_path) << p

    speed = sqrt(u[0] * u[0] + u[1] * u[1])

    S = FunctionSpace(mesh, "CG", 1)
    speed_proj = project(speed, S)

    plt.figure(figsize=(8, 3))
    plot(speed_proj)
    plt.title(f"Speed magnitude: {phase}")
    plt.xlabel("z, m")
    plt.ylabel("r, m")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()