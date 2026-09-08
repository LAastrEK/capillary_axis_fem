import os
import gmsh
import meshio

from dolfin import (
    Mesh,
    XDMFFile,
    MPI,
    MeshFunction,
    SubDomain,
    near,
    cells,
)

from config import Config
from geometry import inner_wall_radii, outer_wall_radii
from ids import (
    ID_AXIS,
    ID_WALL,
    ID_LEFT_OPEN,
    ID_OUTER_OPEN,
    ID_INNER_PORT,
    ID_ANNULAR_PORT,
)


def _wall_surface(occ, cfg: Config, low_func, high_func, z0: float, z1: float):
    """
    Создаёт 2D-поверхность твёрдой стенки в координатах (z, r).
    """
    n = cfg.mesh_points_transition
    zs = [z0 + (z1 - z0) * i / n for i in range(n + 1)]

    pts_low = []
    pts_high = []

    for z in zs:
        r_low = max(0.0, low_func(z))
        r_high = high_func(z)
        pts_low.append(occ.addPoint(z, r_low, 0.0, cfg.lc_global))
        pts_high.append(occ.addPoint(z, r_high, 0.0, cfg.lc_global))

    lines = []

    # Нижняя граница стенки
    for i in range(n):
        lines.append(occ.addLine(pts_low[i], pts_low[i + 1]))

    # Правая перемычка
    lines.append(occ.addLine(pts_low[-1], pts_high[-1]))

    # Верхняя граница стенки, обратно
    for i in range(n, 0, -1):
        lines.append(occ.addLine(pts_high[i], pts_high[i - 1]))

    # Левая перемычка
    lines.append(occ.addLine(pts_high[0], pts_low[0]))

    try:
        loop = occ.addCurveLoop(lines, reorient=True)
    except TypeError:
        loop = occ.addCurveLoop(lines)

    surface = occ.addPlaneSurface([loop])
    return surface


def generate_mesh(cfg: Config):
    """
    Генерирует осесимметричную 2D-геометрию fluid domain:
    прямоугольник (z, r) минус внутренние твёрдые стенки.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("capillary_axisymmetric")

    occ = gmsh.model.occ

    # Fluid bounding box
    rect = occ.addRectangle(
        0.0, 0.0, 0.0,
        cfg.L_domain, cfg.R_domain
    )

    # Стенки начинаются с z_tip и идут до правого края.
    # Немного выносим z1 за L_domain, чтобы boolean cut был чистым.
    z0 = cfg.z_tip
    z1 = cfg.L_domain + 1.0e-6

    inner_wall = _wall_surface(
        occ,
        cfg,
        lambda z: inner_wall_radii(z, cfg)[0],
        lambda z: inner_wall_radii(z, cfg)[1],
        z0,
        z1
    )

    outer_wall = _wall_surface(
        occ,
        cfg,
        lambda z: outer_wall_radii(z, cfg)[0],
        lambda z: outer_wall_radii(z, cfg)[1],
        z0,
        z1
    )

    # Вырезаем стенки из fluid domain
    out, _ = occ.cut(
        [(2, rect)],
        [(2, inner_wall)],
        removeObject=True,
        removeTool=True
    )

    out, _ = occ.cut(
        out,
        [(2, outer_wall)],
        removeObject=True,
        removeTool=True
    )

    occ.synchronize()

    # Mesh algorithm: Frontal-Delaunay обычно хорош для 2D
    gmsh.option.setNumber("Mesh.Algorithm", 6)

    # Background mesh size field: измельчение около кончика
    field_tag = gmsh.model.mesh.field.add("MathEval")

    expr = (
        f"Min({cfg.lc_global}, Max({cfg.lc_nozzle}, "
        f"{cfg.lc_nozzle} + ({cfg.lc_global} - {cfg.lc_nozzle}) * "
        f"Sqrt((x - {cfg.z_tip}) * (x - {cfg.z_tip}) + "
        f"(y - {cfg.nozzle_radius}) * (y - {cfg.nozzle_radius})) / "
        f"{cfg.refine_radius}))"
    )

    gmsh.model.mesh.field.setString(field_tag, "F", expr)
    gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)

    # MSH 2.2 лучше совместим с meshio/FEniCS
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    gmsh.model.mesh.generate(2)
    gmsh.write(cfg.mesh_msh)
    gmsh.finalize()

    _convert_msh_to_xdmf(cfg.mesh_msh, cfg.mesh_xdmf)


def _convert_msh_to_xdmf(msh_path: str, xdmf_path: str):
    """
    Конвертирует Gmsh .msh в XDMF для FEniCS.
    Берёт только треугольники.
    """
    mesh = meshio.read(msh_path)

    points = mesh.points[:, :2]

    triangles = None
    for block in mesh.cells:
        if block.type == "triangle":
            triangles = block.data
            break

    if triangles is None:
        raise RuntimeError("В mesh-файле нет треугольников.")

    meshio.write(
        xdmf_path,
        meshio.Mesh(
            points=points,
            cells=[("triangle", triangles)]
        )
    )


def read_mesh(cfg: Config) -> Mesh:
    """
    Читает XDMF mesh в FEniCS.
    """
    mesh = Mesh()
    with XDMFFile(MPI.comm_world, cfg.mesh_xdmf) as f:
        f.read(mesh)
    return mesh


class _AllBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class _Axis(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, 1.0e-9)


class _LeftOpen(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, 1.0e-9)


class _OuterOpen(SubDomain):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], self.cfg.R_domain, 1.0e-9)


class _InnerPort(SubDomain):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def inside(self, x, on_boundary):
        tol = 1.0e-9
        return (
            on_boundary
            and near(x[0], self.cfg.L_domain, tol)
            and x[1] < self.cfg.inner_body_radius - tol
        )


class _AnnularPort(SubDomain):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def inside(self, x, on_boundary):
        tol = 1.0e-9
        r_inner_wall_high = self.cfg.inner_body_radius + self.cfg.wall_thickness
        r_outer_wall_low = self.cfg.outer_body_inner

        return (
            on_boundary
            and near(x[0], self.cfg.L_domain, tol)
            and x[1] > r_inner_wall_high + tol
            and x[1] < r_outer_wall_low - tol
        )


def mark_boundaries(mesh: Mesh, cfg: Config):
    """
    Размечает границы:
    - все внешние грани сначала считаются стенками;
    - затем выделяются axis, left open, outer open, inner port, annular port.
    """
    boundaries = MeshFunction(
        "size_t",
        mesh,
        mesh.topology().dim() - 1,
        0
    )

    _AllBoundary().mark(boundaries, ID_WALL)
    _Axis().mark(boundaries, ID_AXIS)
    _LeftOpen().mark(boundaries, ID_LEFT_OPEN)
    _OuterOpen(cfg).mark(boundaries, ID_OUTER_OPEN)
    _InnerPort(cfg).mark(boundaries, ID_INNER_PORT)
    _AnnularPort(cfg).mark(boundaries, ID_ANNULAR_PORT)

    return boundaries


def mark_cells(mesh: Mesh, cfg: Config):
    """
    Размечает ячейки:
    - working zone: зона перед кончиком, z < z_tip;
    - target zone: зона предполагаемого воздействия на клетки.
    """
    working_markers = MeshFunction(
        "size_t",
        mesh,
        mesh.topology().dim(),
        0
    )

    target_markers = MeshFunction(
        "size_t",
        mesh,
        mesh.topology().dim(),
        0
    )

    z_target_left = cfg.z_tip - cfg.target_length

    for c in cells(mesh):
        mid = c.midpoint()
        x = mid.x()
        y = mid.y()

        if x < cfg.z_tip:
            working_markers[c.index()] = 1

        if z_target_left <= x <= cfg.z_tip and y <= cfg.target_radius:
            target_markers[c.index()] = 1

    return working_markers, target_markers
