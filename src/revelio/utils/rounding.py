from decimal import ROUND_HALF_UP, Decimal


def round_half_up(n: float) -> int:
    return int(Decimal(n).to_integral_value(ROUND_HALF_UP))
