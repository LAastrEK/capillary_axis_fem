import math
from config import Config


def smooth_factor(z: float, cfg: Config) -> float:
    """
    Плавный tanh-переход от кончика к телу капилляра.
    0 - у кончика, 1 - в теле.
    """
    if z <= cfg.z_tip:
        return 0.0

    z_end = cfg.z_tip + cfg.L_transition

    if z >= z_end:
        return 1.0

    t = (z - cfg.z_tip) / cfg.L_transition
    denom = math.tanh(cfg.sharpness * 0.5)

    if abs(denom) < 1.0e-12:
        return t

    val = 0.5 * (1.0 + math.tanh(cfg.sharpness * (t - 0.5)) / denom)
    return float(max(0.0, min(1.0, val)))


def inner_open_radius(z: float, cfg: Config) -> float:
    """
    Радиус внутреннего открытого канала.
    """
    f = smooth_factor(z, cfg)
    return cfg.nozzle_radius + (cfg.inner_body_radius - cfg.nozzle_radius) * f


def inner_wall_radii(z: float, cfg: Config):
    """
    Возвращает (r_low, r_high) внутренней твёрдой стенки.
    Открытый канал находится внутри r_low.
    """
    r_open = inner_open_radius(z, cfg)
    r_low = r_open
    r_high = r_open + cfg.wall_thickness
    return r_low, r_high


def outer_wall_radii(z: float, cfg: Config):
    """
    Возвращает (r_low, r_high) внешней твёрдой стенки.
    Annular channel находится между inner_wall_high и outer_wall_low.
    """
    f = smooth_factor(z, cfg)
    r_low = cfg.outer_tip_inner + (cfg.outer_body_inner - cfg.outer_tip_inner) * f
    r_high = r_low + cfg.wall_thickness
    return r_low, r_high