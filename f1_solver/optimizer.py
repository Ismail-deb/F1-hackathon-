"""Strategy optimizer using scipy differential_evolution.

The strategy is encoded as a flat float vector:
  [speed_factor_0..N, brake_factor_0..N,        # per straight, per lap  (if same per lap)
   pit_binary_0..L,                              # 0/1 per lap
   tyre_idx_0..P,                               # tyre index per stint
   fuel_0..P]                                   # refuel per pit stop

For tractability the per-straight decisions are replicated across laps
(one set of values reused every lap) unless the track is very short.

The encoded vector is decoded → Strategy → simulate_race → score.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from .models import (
    LapStrategy,
    RaceConfig,
    Segment,
    StraightStrategy,
    Strategy,
    TyreSet,
)
from .physics import calc_max_corner_speed, calc_tyre_friction
from .scorer import calc_score
from .simulator import simulate_race


# ── Helpers ──────────────────────────────────────────────────────────

def _max_speed_before_corner(
    seg: Segment,
    segments: List[Segment],
    tyre: TyreSet,
    car_crawl: float,
) -> float:
    """Upper-bound target speed: governed by the corner after this straight."""
    seg_pos = segments.index(seg)
    for ns in segments[seg_pos + 1:]:
        if ns.seg_type == "corner":
            friction = calc_tyre_friction(tyre.base_friction, 0.0, 1.0)
            return calc_max_corner_speed(ns.radius, friction, car_crawl)
    return 200.0   # no corner after — open end of lap


def _build_strategy(
    x: np.ndarray,
    config: RaceConfig,
    n_straights: int,
    pit_laps: List[int],
    stint_tyres: List[int],
    prev_pit_speeds: Optional[List[float]] = None,
) -> Strategy:
    """Decode parameter vector x into a Strategy object."""
    segments = config.track.segments
    car = config.car
    total_laps = config.track.total_laps

    # x layout:
    #   [0 .. n_straights)          speed_factor per straight (0..1)
    #   [n_straights .. 2*n_str)    brake_factor per straight (0..1)
    #   [2*n_str .. 2*n_str + n_pit) fuel per pit (only level 2+)
    n_str = n_straights
    speed_factors = x[:n_str]
    brake_factors = x[n_str: 2 * n_str]

    # Fuel per pit stop
    n_pit = len(pit_laps)
    fuel_amounts: List[float] = []
    if config.level >= 2 and n_pit > 0:
        fuel_amounts = list(x[2 * n_str: 2 * n_str + n_pit])

    # Build per-lap strategies
    straights = [s for s in segments if s.seg_type == "straight"]
    lap_strategies: List[LapStrategy] = []
    stint_idx = 0  # which tyre stint we're in
    initial_tyre_id = config.tyre_sets[stint_tyres[0]].id

    for lap_idx in range(total_laps):
        straight_strats: List[StraightStrategy] = []
        for s_idx, seg in enumerate(straights):
            sf = float(np.clip(speed_factors[s_idx], 0.0, 1.0))
            bf = float(np.clip(brake_factors[s_idx], 0.0, 1.0))

            # Upper speed bound for this straight
            tyre = config.tyre_sets[stint_tyres[min(stint_idx, len(stint_tyres) - 1)]]
            max_spd = _max_speed_before_corner(seg, segments, tyre, car.crawl_constant)
            target_speed = sf * max_spd

            # Brake start: fraction of segment length dedicated to braking
            brake_start = bf * seg.length

            straight_strats.append(StraightStrategy(target_speed, brake_start))

        # Pit decisions
        is_pit_lap = lap_idx + 1 in pit_laps  # laps are 1-indexed for strategy
        pit_lap_order = pit_laps.index(lap_idx + 1) if is_pit_lap else -1

        refuel = 0.0
        new_tyre_id: Optional[int] = None
        if is_pit_lap:
            stint_idx += 1
            if stint_idx < len(stint_tyres):
                new_tyre_id = config.tyre_sets[stint_tyres[stint_idx]].id
            if config.level >= 2 and pit_lap_order < len(fuel_amounts):
                refuel = float(np.clip(fuel_amounts[pit_lap_order], 0.0, car.fuel_capacity))

        lap_strategies.append(LapStrategy(
            straight_strategies=straight_strats,
            pit_enter=is_pit_lap,
            tyre_change_set_id=new_tyre_id,
            fuel_refuel_amount=refuel,
        ))

    return Strategy(initial_tyre_id=initial_tyre_id, laps=lap_strategies)


def _objective(
    x: np.ndarray,
    config: RaceConfig,
    n_straights: int,
    pit_laps: List[int],
    stint_tyres: List[int],
) -> float:
    """Negative score (minimise → maximise score)."""
    try:
        strat = _build_strategy(x, config, n_straights, pit_laps, stint_tyres)
        result = simulate_race(config, strat)
        score = calc_score(config, result)
        return -score
    except Exception:
        return 1e12


def _build_bounds(
    config: RaceConfig,
    n_straights: int,
    n_pit: int,
) -> List[Tuple[float, float]]:
    """Build bounds for the DE parameter vector."""
    bounds = []
    # speed_factor per straight: (0.3, 1.0) — don't go too slow
    bounds += [(0.3, 1.0)] * n_straights
    # brake_factor per straight: (0.05, 0.9) — fraction of straight used for braking
    bounds += [(0.05, 0.9)] * n_straights
    # fuel per pit stop
    if config.level >= 2:
        bounds += [(0.0, config.car.fuel_capacity)] * n_pit
    return bounds


def _choose_pit_schedule(
    config: RaceConfig,
    n_stops: int,
) -> List[int]:
    """Evenly space pit stops across laps. Returns sorted list of lap numbers (1-indexed)."""
    total = config.track.total_laps
    if n_stops == 0:
        return []
    interval = total / (n_stops + 1)
    return sorted(set(int(round(interval * (i + 1))) for i in range(n_stops)))


def _choose_stint_tyres(config: RaceConfig, n_stints: int) -> List[int]:
    """Choose tyre indices for each stint.

    Returns indices into config.tyre_sets.
    Level 1-2: best available per stint (softest compound).
    Level 3-4: attempt to use different compounds for strategy variety.
    """
    n_sets = len(config.tyre_sets)
    if n_sets == 0:
        return [0] * n_stints

    # Sort by base_friction descending (softest first)
    sorted_idx = sorted(range(n_sets), key=lambda i: -config.tyre_sets[i].base_friction)

    result = []
    for i in range(n_stints):
        result.append(sorted_idx[i % len(sorted_idx)])
    return result


def optimize(config: RaceConfig, time_budget_s: float = 60.0) -> Strategy:
    """Run differential evolution and return the best Strategy found."""
    segments = config.track.segments
    n_straights = sum(1 for s in segments if s.seg_type == "straight")
    total_laps = config.track.total_laps

    best_score = -1e18
    best_strategy: Optional[Strategy] = None

    # Try different numbers of pit stops
    for n_stops in _get_stop_range(config):
        pit_laps = _choose_pit_schedule(config, n_stops)
        stint_tyres = _choose_stint_tyres(config, n_stops + 1)
        n_pit = len(pit_laps)

        bounds = _build_bounds(config, n_straights, n_pit)

        def obj(x):
            return _objective(x, config, n_straights, pit_laps, stint_tyres)

        result = differential_evolution(
            obj,
            bounds,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            seed=42,
            workers=1,   # safe default; set to -1 for multicore if pickling works
            polish=True,
            mutation=(0.5, 1.5),
            recombination=0.9,
        )

        strat = _build_strategy(
            result.x, config, n_straights, pit_laps, stint_tyres
        )
        sim_result = simulate_race(config, strat)
        score = calc_score(config, sim_result)

        if score > best_score:
            best_score = score
            best_strategy = strat

    if best_strategy is None:
        # Fallback: naive strategy
        best_strategy = _naive_strategy(config)

    return best_strategy


def _get_stop_range(config: RaceConfig) -> range:
    """Returns the range of pit-stop counts to try."""
    if config.level == 1:
        return range(0, 1)   # no pit needed for L1
    total = config.track.total_laps
    max_stops = min(total - 1, 4)
    return range(0, max_stops + 1)


def _naive_strategy(config: RaceConfig) -> Strategy:
    """Build a conservative strategy without optimization."""
    segments = config.track.segments
    straights = [s for s in segments if s.seg_type == "straight"]
    tyre = config.tyre_sets[0] if config.tyre_sets else None
    initial_tyre_id = tyre.id if tyre else 0

    lap_strategies = []
    for _ in range(config.track.total_laps):
        strats = []
        for seg in straights:
            max_spd = _max_speed_before_corner(
                seg, segments, tyre, config.car.crawl_constant
            ) if tyre else 50.0
            strats.append(StraightStrategy(
                target_speed=max_spd * 0.8,
                brake_start=seg.length * 0.3,
            ))
        lap_strategies.append(LapStrategy(
            straight_strategies=strats,
            pit_enter=False,
        ))

    return Strategy(initial_tyre_id=initial_tyre_id, laps=lap_strategies)
