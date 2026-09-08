from config import Config
from ids import ID_INNER_PORT, ID_ANNULAR_PORT


def get_active_bc(phase: str, cfg: Config):
    """
    Возвращает ID активного порта и вектор скорости.

    injection:
        поток входит через внутренний канал справа налево,
        поэтому u_z отрицательная.

    aspiration:
        поток входит/выходит через annulus справа налево или слева направо,
        здесь используем положительное направление u_z.
    """
    if phase == "injection":
        return ID_INNER_PORT, (-cfg.u_inj, 0.0)

    if phase == "aspiration":
        return ID_ANNULAR_PORT, (cfg.u_asp, 0.0)

    raise ValueError(f"Unknown phase: {phase}")