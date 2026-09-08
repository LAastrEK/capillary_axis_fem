import numpy as np

from dolfin import (
    RectangleMesh,
    Point,
    FunctionSpace,
    Function,
    MixedElement,
    VectorElement,
    FiniteElement,
    TrialFunctions,
    TestFunctions,
    SpatialCoordinate,
    Measure,
    DirichletBC,
    Constant,
    conditional,
    lt,
    inner,
    as_tensor,
    assemble_system,
    solve,
    MeshFunction,
    SubDomain,
    near,
    project,
    pi,
)


def run_poiseuille_verification():
    """
    Верификация осесимметричного Stokes на течении Пуазёйля.

    Область:
        z in [0, L]
        r in [0, R]

    Граничные условия:
        r = 0: u_r = 0
        r = R: u = 0
        z = 0: p = dp
        z = L: p = 0

    Аналитическое решение:
        u_z(r) = dp / (4 * mu * L) * (R^2 - r^2)
    """

    L = 2.0e-3
    R = 100.0e-6
    rho = 1000.0
    nu = 1.0e-6
    mu = rho * nu
    dp = 1.0

    mesh = RectangleMesh(
        Point(0.0, 0.0),
        Point(L, R),
        120,
        24
    )

    cell = mesh.ufl_cell()

    P2 = VectorElement("CG", cell, 2)
    P1 = FiniteElement("CG", cell, 1)

    W = FunctionSpace(mesh, MixedElement([P2, P1]))

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    r = SpatialCoordinate(mesh)[1]
    eps = 1.0e-12
    r_safe = conditional(lt(r, eps), eps, r)

    def epsilon_axisymmetric(uu):
        uz = uu[0]
        ur = uu[1]

        e_zz = uz.dx(0)
        e_rr = ur.dx(1)
        e_zr = 0.5 * (uz.dx(1) + ur.dx(0))
        e_tt = conditional(lt(r, eps), 0.0, ur / r_safe)

        return as_tensor([
            [e_zz, e_zr, 0.0],
            [e_zr, e_rr, 0.0],
            [0.0, 0.0, e_tt]
        ])

    dx = Measure("dx", domain=mesh)
    area = 2.0 * pi * r_safe

    div_v_weighted = (
        r * v[0].dx(0)
        + r * v[1].dx(1)
        + v[1]
    )

    div_u_weighted = (
        r * u[0].dx(0)
        + r * u[1].dx(1)
        + u[1]
    )

    a = (
        2.0 * mu * inner(epsilon_axisymmetric(u), epsilon_axisymmetric(v)) * area * dx
        - p * div_v_weighted * 2.0 * pi * dx
        - q * div_u_weighted * 2.0 * pi * dx
    )

    L_form = inner(Constant((0.0, 0.0)), v) * area * dx

    boundaries = MeshFunction(
        "size_t",
        mesh,
        mesh.topology().dim() - 1,
        0
    )

    class Wall(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], R, 1.0e-9)

    class Axis(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0.0, 1.0e-9)

    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0.0, 1.0e-9)

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], L, 1.0e-9)

    Wall().mark(boundaries, 1)
    Axis().mark(boundaries, 2)
    Left().mark(boundaries, 3)
    Right().mark(boundaries, 4)

    bcs = [
        DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 1),
        DirichletBC(W.sub(0).sub(1), 0.0, boundaries, 2),
        DirichletBC(W.sub(1), dp, boundaries, 3),
        DirichletBC(W.sub(1), 0.0, boundaries, 4),
    ]

    A, b = assemble_system(a, L_form, bcs)

    w = Function(W)
    solve(A, w.vector(), b)

    u_sol, p_sol = w.split()
    uz_sol = u_sol.sub(0)

    V1 = FunctionSpace(mesh, "CG", 1)
    uz_proj = project(uz_sol, V1)

    coords = V1.tabulate_dof_coordinates().reshape((-1, 2))
    vals = uz_proj.vector().get_local()

    mask = np.abs(coords[:, 0] - 0.5 * L) < 0.05 * L

    r_vals = coords[mask, 1]
    uz_num = vals[mask]

    uz_exact = dp / (4.0 * mu * L) * (R * R - r_vals * r_vals)

    denom = max(np.max(np.abs(uz_exact)), 1.0e-30)
    rel_l2 = np.sqrt(np.mean((uz_num - uz_exact) ** 2)) / denom

    print("Poiseuille axisymmetric verification")
    print(f"max exact velocity = {np.max(uz_exact):.6e} m/s")
    print(f"max numeric velocity = {np.max(uz_num):.6e} m/s")
    print(f"relative L2 error = {rel_l2:.6e}")

    return rel_l2