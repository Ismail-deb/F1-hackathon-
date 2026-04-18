"""Scoring functions per competition level."""
from __future__ import annotations

from .models import RaceConfig, SimResult


def calc_score(config: RaceConfig, result: SimResult) -> float:
    """Compute the final score for a simulation result."""
    if result.total_time <= 0:
        return 0.0

    # ── Base score (all levels) ───────────────────────────────────
    base = 500_000.0 * (config.time_reference / result.total_time) ** 3

    if config.level == 1:
        return base

    # ── Fuel bonus (Level 2+) ─────────────────────────────────────
    soft_cap = config.fuel_soft_cap
    fuel_bonus = 0.0
    if soft_cap > 0:
        ratio = result.fuel_used / soft_cap
        fuel_bonus = -500_000.0 * (1.0 - ratio) ** 2 + 500_000.0

    if config.level in (2, 3):
        return base + fuel_bonus

    # ── Tyre bonus (Level 4) ──────────────────────────────────────
    total_degradation = sum(u.total_degradation for u in result.tyre_usages)
    tyre_bonus = 100_000.0 * total_degradation - 50_000.0 * result.num_blowouts

    return base + fuel_bonus + tyre_bonus
