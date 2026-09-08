import numpy as np

from dolfin import (
    FunctionSpace,
    Function,
    SpatialCoordinate,
    Measure,
    conditional,
    lt,
    sqrt,
    assemble,
    project,
    Constant,
    pi,
)

from config import Config
from ids import (
    ID_INNER_PORT,
    ID_ANNULAR_PORT,
    ID_LEFT_OPEN,
)


def compute_metrics(
    w: Function,
    mesh,
    boundaries,
    working_markers,
    target_markers,
    cfg: Config,
    phase: str
):
    """
    Считает физические метрики:
    - shear rate magnitude;
    - vorticity;
    - stagnation area fraction;
    - flow rates;
    - average pressures.
    """

    u, p = w.split()

    r = SpatialCoordinate(mesh)[1]
    eps = 1.0e-12
    r_safe = conditional(lt(r, eps), eps, r)

    area_weight = 2.0 * pi * r_safe

    uz = u[0]
    ur = u[1]

    speed = sqrt(uz * uz + ur * ur)

    # Осесимметричный shear rate magnitude
    e_zz = uz.dx(0)
    e_rr = ur.dx(1)
    e_zr = 0.5 * (uz.dx(1) + ur.dx(0))
    e_tt = conditional(lt(r, eps), 0.0, ur / r_safe)

    gamma_dot = sqrt(
        2.0 * (
            e_zz * e_zz
            + e_rr * e_rr
            + e_tt * e_tt
            + 2.0 * e_zr * e_zr
        )
    )

    # Осесимметричная завихренность, theta-компонента
    omega = ur.dx(0) - uz.dx(1)
    omega_abs = sqrt(omega * omega)

    dx_working = Measure(
        "dx",
        domain=mesh,
        subdomain_data=working_markers
    )(1)

    dx_target = Measure(
        "dx",
        domain=mesh,
        subdomain_data=target_markers
    )(1)

    area_working = assemble(area_weight * dx_working)
    area_target = assemble(area_weight * dx_target)

    # Средний |omega| в рабочей зоне
    if area_working > 1.0e-30:
        omega_mean = assemble(omega_abs * area_weight * dx_working) / area_working
    else:
        omega_mean = 0.0

    # Проекции для max / percentile
    DG1 = FunctionSpace(mesh, "DG", 1)

    speed_proj = project(speed, DG1)
    gamma_proj = project(gamma_dot, DG1)
    omega_proj = project(omega_abs, DG1)

    coords = DG1.tabulate_dof_coordinates().reshape((-1, 2))

    speed_vals = speed_proj.vector().get_local()
    gamma_vals = gamma_proj.vector().get_local()
    omega_vals = omega_proj.vector().get_local()

    # Masks по координатам DOF
    working_mask = coords[:, 0] < cfg.z_tip

    target_mask = (
        (coords[:, 0] >= cfg.z_tip - cfg.target_length)
        & (coords[:, 0] <= cfg.z_tip)
        & (coords[:, 1] <= cfg.target_radius)
    )

    # U_ref для stagnation threshold
    if np.any(working_mask):
        U_ref = float(np.percentile(speed_vals[working_mask], 95))
    else:
        U_ref = 0.0

    # Физический нижний порог 10 мкм/с
    U_threshold = max(0.05 * U_ref, 1.0e-5)

    # Площадь застойных зон
    stagnant_area = assemble(
        conditional(lt(speed, U_threshold), 1.0, 0.0)
        * area_weight
        * dx_working
    )

    if area_working > 1.0e-30:
        k_stag = 100.0 * stagnant_area / area_working
    else:
        k_stag = 0.0

    # Максимальный сдвиг в целевой зоне
    if np.any(target_mask):
        gamma_max_target = float(np.max(gamma_vals[target_mask]))
    else:
        gamma_max_target = 0.0

    # 99-й перцентиль завихренности в рабочей зоне
    if np.any(working_mask):
        omega_p99 = float(np.percentile(omega_vals[working_mask], 99))
    else:
        omega_p99 = 0.0

    # Расходы через порты
    ds = Measure(
        "ds",
        domain=mesh,
        subdomain_data=boundaries
    )

    Q_inner = assemble(2.0 * pi * r * uz * ds(ID_INNER_PORT))
    Q_annular = assemble(2.0 * pi * r * uz * ds(ID_ANNULAR_PORT))
    Q_left = assemble(2.0 * pi * r * uz * ds(ID_LEFT_OPEN))

    def average_pressure(boundary_id: int) -> float:
        area_b = assemble(Constant(1.0) * ds(boundary_id))
        if area_b < 1.0e-30:
            return 0.0
        return assemble(p * ds(boundary_id)) / area_b

    p_inner = average_pressure(ID_INNER_PORT)
    p_annular = average_pressure(ID_ANNULAR_PORT)
    p_left = average_pressure(ID_LEFT_OPEN)

    return {
        "phase": phase,
        "area_working_m2": float(area_working),
        "area_target_m2": float(area_target),
        "U_ref_m_s": float(U_ref),
        "U_threshold_m_s": float(U_threshold),
        "gamma_max_target_1_s": float(gamma_max_target),
        "omega_mean_working_1_s": float(omega_mean),
        "omega_p99_working_1_s": float(omega_p99),
        "k_stag_percent": float(k_stag),
        "Q_inner_m3_s": float(Q_inner),
        "Q_annular_m3_s": float(Q_annular),
        "Q_left_m3_s": float(Q_left),
        "p_inner_Pa": float(p_inner),
        "p_annular_Pa": float(p_annular),
        "p_left_Pa": float(p_left),
    }