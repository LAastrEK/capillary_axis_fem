def reynolds(U: float, D: float, nu: float) -> float:
    """
    Re = U * D / nu
    U: м/с
    D: м
    nu: м^2/с
    """
    return abs(U) * D / nu