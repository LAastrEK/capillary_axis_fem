import argparse
import json
import os

from config import Config
from meshing import (
    generate_mesh,
    read_mesh,
    mark_boundaries,
    mark_cells,
)
from solver import AxisymmetricStokesSolver
from metrics import compute_metrics
from postprocess import save_phase_results
from verification import run_poiseuille_verification
from units import reynolds


def main():
    parser = argparse.ArgumentParser(
        description="2D axisymmetric FEM capillary injection/aspiration model"
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run Poiseuille verification and exit"
    )

    parser.add_argument(
        "--remesh",
        action="store_true",
        help="Force mesh regeneration"
    )

    args = parser.parse_args()

    cfg = Config()

    if args.verify:
        run_poiseuille_verification()
        return

    if args.remesh or not os.path.exists(cfg.mesh_xdmf):
        print("Generating mesh...")
        generate_mesh(cfg)
    else:
        print("Using existing mesh. Use --remesh to regenerate.")

    print("Reading mesh...")
    mesh = read_mesh(cfg)

    print("Marking boundaries and cells...")
    boundaries = mark_boundaries(mesh, cfg)
    working_markers, target_markers = mark_cells(mesh, cfg)

    solver = AxisymmetricStokesSolver(mesh, boundaries, cfg)

    results = {}

    phases = ["injection", "aspiration"]

    for phase in phases:
        print(f"Solving phase: {phase}")
        w = solver.solve(phase)

        print(f"Computing metrics: {phase}")
        metrics = compute_metrics(
            w,
            mesh,
            boundaries,
            working_markers,
            target_markers,
            cfg,
            phase
        )

        if phase == "injection":
            U_char = cfg.u_inj
        else:
            U_char = cfg.u_asp

        metrics["Re_nozzle"] = reynolds(
            U_char,
            2.0 * cfg.nozzle_radius,
            cfg.nu
        )

        results[phase] = metrics

        print(f"Saving results: {phase}")
        save_phase_results(phase, w, mesh, cfg, metrics)

    metrics_path = os.path.join(cfg.output_dir, "metrics.json")

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nFinal metrics:")
    print(json.dumps(results, indent=2))
    print(f"\nResults saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()