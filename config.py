from dataclasses import dataclass, field
import os


@dataclass
class Config:
    """
    Все величины в SI:
    длина - м,
    скорость - м/с,
    вязкость - м^2/с,
    плотность - кг/м^3.
    """

    # Расчётная область
    L_domain: float = 4.0e-3       # 4 мм
    R_domain: float = 0.8e-3       # 0.8 мм

    # Положение кончика
    z_tip: float = 1.0e-3          # кончик находится на z = 1 мм

    # Переход сопла
    L_transition: float = 2.0e-3   # длина плавного перехода
    sharpness: float = 2.0         # крутизна tanh-перехода

    # Геометрия капилляра
    nozzle_radius: float = 50.0e-6        # 50 мкм радиус сопла
    wall_thickness: float = 20.0e-6       # 20 мкм толщина стенки
    inner_body_radius: float = 200.0e-6   # внутренний канал в теле капилляра

    # Внешняя стенка / annulus
    outer_tip_inner: float = 150.0e-6     # внутренний радиус внешней стенки у кончика
    outer_body_inner: float = 400.0e-6    # внутренний радиус внешней стенки в теле

    # Жидкость
    rho: float = 1000.0            # кг/м^3
    nu: float = 1.0e-6             # м^2/с, примерно вода

    # Протокол
    u_inj: float = 5.0e-3          # 5 мм/с инжекция
    u_asp: float = 5.0e-3          # 5 мм/с аспирация

    # Целевая зона клеток
    target_length: float = 300.0e-6   # зона перед кончиком
    target_radius: float = 200.0e-6   # радиус целевой зоны

    # Сетка
    lc_global: float = 60.0e-6        # глобальный размер ячейки
    lc_nozzle: float = 5.0e-6         # размер ячейки у кончика
    refine_radius: float = 250.0e-6   # радиус зоны измельчения
    mesh_points_transition: int = 80

    # Вывод
    output_dir: str = "output"

    # Служебные пути
    mesh_xdmf: str = field(init=False)
    mesh_msh: str = field(init=False)

    # Производные размеры
    outer_tip_outer: float = field(init=False)
    outer_body_outer: float = field(init=False)

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

        self.mesh_xdmf = os.path.join(self.output_dir, "mesh.xdmf")
        self.mesh_msh = os.path.join(self.output_dir, "mesh.msh")

        self.outer_tip_outer = self.outer_tip_inner + self.wall_thickness
        self.outer_body_outer = self.outer_body_inner + self.wall_thickness

        if self.outer_body_outer >= self.R_domain:
            raise ValueError(
                "R_domain должен быть больше outer_body_outer. "
                "Увеличьте R_domain или уменьшите внешний радиус."
            )

        if self.nozzle_radius + self.wall_thickness >= self.outer_tip_inner:
            raise ValueError(
                "Annular gap у кончика отрицательный. "
                "Увеличьте outer_tip_inner или уменьшите nozzle_radius/wall_thickness."
            )