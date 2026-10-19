"""Her personality dials: the bands each one lives in, and the nudge a single request gives."""
DIALS = ("sweetness", "teasing", "clinginess", "verbosity", "emoji")
BANDS: dict[str, tuple[float, float]] = {"sweetness": (0.6, 1.0), "teasing": (0.1, 0.8), "clinginess": (0.4, 1.0),
                                         "verbosity": (0.15, 0.7), "emoji": (0.0, 0.6)}
DELTA = 0.15  # one explicit request


def clamp(dial: str, value: float) -> float:
    """Keep a dial inside its band (rounded to 3 decimals); an unknown dial is a bug."""
    if dial not in BANDS:
        raise ValueError(f"unknown dial: {dial!r}")
    lo, hi = BANDS[dial]
    return round(min(hi, max(lo, value)), 3)
