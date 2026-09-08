from dolfin import (
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
    pi,
)

from config import Config
from ids import ID_AXIS, ID_WALL
from protocol import get_active_bc


class AxisymmetricStokesSolver:
    """
    Осесимметричный стационарный Stokes.

    Координаты:
        x[0] = z
        x[1] = r

    Скорость:
        u[0] = u_z
        u[1] = u_r
    """

    def __init__(self, mesh, boundaries, cfg: Config):
        self.mesh = mesh
        self.boundaries = boundaries
        self.cfg = cfg

        cell = mesh.ufl_cell()

        P2 = VectorElement("CG", cell, 2)
        P1 = FiniteElement("CG", cell, 1)

        self.W = FunctionSpace(mesh, MixedElement([P2, P1]))

        self.mu = Constant(cfg.rho * cfg.nu)

        self.r = SpatialCoordinate(mesh)[1]
        self.eps = 1.0e-12
        self.r_safe = conditional(lt(self.r, self.eps), self.eps, self.r)

    def _epsilon_axisymmetric(self, u):
        """
        Осесимметричный тензор деформации.

        Компоненты:
            e_zz = du_z/dz
            e_rr = du_r/dr
            e_zr = 0.5 * (du_z/dr + du_r/dz)
            e_tt = u_r / r
        """
        uz = u[0]
        ur = u[1]

        e_zz = uz.dx(0)
        e_rr = ur.dx(1)
        e_zr = 0.5 * (uz.dx(1) + ur.dx(0))
        e_tt = conditional(lt(self.r, self.eps), 0.0, ur / self.r_safe)

        return as_tensor([
            [e_zz, e_zr, 0.0],
            [e_zr, e_rr, 0.0],
            [0.0, 0.0, e_tt]
        ])

    def solve(self, phase: str) -> Function:
        """
        Решает стационарный Stokes для заданной фазы:
        - injection
        - aspiration
        """
        u, p = TrialFunctions(self.W)
        v, q = TestFunctions(self.W)

        dx = Measure("dx", domain=self.mesh)

        eps_u = self._epsilon_axisymmetric(u)
        eps_v = self._epsilon_axisymmetric(v)

        area = 2.0 * pi * self.r_safe

        # Взвешенная осесимметричная дивергенция:
        # div(u) = du_z/dz + du_r/dr + u_r/r
        # r * div(u) = r*du_z/dz + r*du_r/dr + u_r
        div_v_weighted = (
            self.r * v[0].dx(0)
            + self.r * v[1].dx(1)
            + v[1]
        )

        div_u_weighted = (
            self.r * u[0].dx(0)
            + self.r * u[1].dx(1)
            + u[1]
        )

        a = (
            2.0 * self.mu * inner(eps_u, eps_v) * area * dx
            - p * div_v_weighted * 2.0 * pi * dx
            - q * div_u_weighted * 2.0 * pi * dx
        )

        L = inner(Constant((0.0, 0.0)), v) * area * dx

        bcs = []

        # No-slip walls
        bcs.append(
            DirichletBC(
                self.W.sub(0),
                Constant((0.0, 0.0)),
                self.boundaries,
                ID_WALL
            )
        )

        # Axisymmetry: u_r = 0 at r = 0
        bcs.append(
            DirichletBC(
                self.W.sub(0).sub(1),
                0.0,
                self.boundaries,
                ID_AXIS
            )
        )

        # Active velocity port
        port_id, velocity = get_active_bc(phase, self.cfg)
        bcs.append(
            DirichletBC(
                self.W.sub(0),
                Constant(velocity),
                self.boundaries,
                port_id
            )
        )

        # Pressure gauge: fix pressure at one point
        # Обычно точка (0, 0) присутствует на оси и левом выходе.
        bcs.append(
            DirichletBC(
                self.W.sub(1),
                0.0,
                "near(x[0], 0.0) && near(x[1], 0.0)",
                method="pointwise"
            )
        )

        A, b = assemble_system(a, L, bcs)

        w = Function(self.W)
        solve(A, w.vector(), b)

        return w