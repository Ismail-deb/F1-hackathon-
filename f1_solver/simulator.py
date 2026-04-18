"""Full race simulator.

simulate_race(config, strategy) -> SimResult

Is a pure function (no side effects) that can be called thousands of
times by the optimizer.
"""
from __future__ import annotations

import math
from typing import List, Optional

from .models import (
    Car,
    RaceConfig,
    Segment,
    SimResult,
    Strategy,
    Track,
    TyreSet,
    TyreUsage,
    WeatherCondition,
)
from .physics import (
    calc_corner_degradation,
    calc_cruise_time,
    calc_fuel_consumption,
    calc_max_corner_speed,
    calc_pit_stop_time,
    calc_tyre_friction,
    simulate_straight,
)


def _get_weather(conditions: List[WeatherCondition], elapsed: float) -> WeatherCondition:
    """Return the active weather condition at the given elapsed race time.

    Cycles back to the beginning if the race outlasts all defined conditions.
    """
    if not conditions:
        return WeatherCondition(0.0, 1.0, 1.0, 1.0, 1.0)

    total_span = conditions[-1].start_time
    if total_span <= 0:
        return conditions[0]

    # Cycle elapsed time back into the defined window
    t = elapsed % total_span if total_span > 0 else elapsed

    active = conditions[0]
    for cond in conditions:
        if cond.start_time <= t:
            active = cond
        else:
            break
    return active


def _find_tyre(tyre_sets: List[TyreSet], tyre_id: int) -> TyreSet:
    for t in tyre_sets:
        if t.id == tyre_id:
            return t
    raise ValueError(f"Tyre set id={tyre_id} not found")


def simulate_race(config: RaceConfig, strategy: Strategy) -> SimResult:
    """Simulate a full race and return the result."""
    car: Car = config.car
    track: Track = config.track
    segments: List[Segment] = track.segments

    # ── State tracking ───────────────────────────────────────────
    elapsed = 0.0
    fuel = config.initial_fuel
    total_fuel_used = 0.0

    # Active tyre
    current_tyre = _find_tyre(config.tyre_sets, strategy.initial_tyre_id)
    tyre_degradation: dict[int, float] = {t.id: 0.0 for t in config.tyre_sets}
    tyre_blowout: dict[int, bool] = {t.id: False for t in config.tyre_sets}
    tyres_used: list[int] = [strategy.initial_tyre_id]

    is_limp = False
    current_speed = car.start_speed

    # Identify straights in order for indexing
    straight_ids = [s.id for s in segments if s.seg_type == "straight"]

    for lap_idx, lap_strat in enumerate(strategy.laps):
        straight_strat_idx = 0  # index into lap_strat.straight_strategies

        for seg in segments:
            weather = _get_weather(config.weather, elapsed)
            tyre_deg = tyre_degradation[current_tyre.id]
            tyre_friction = calc_tyre_friction(
                current_tyre.base_friction,
                tyre_deg,
                weather.friction_multiplier,
            )
            eff_deg_rate = current_tyre.degradation_rate * weather.degradation_rate_multiplier

            # Check limp mode: fuel empty or tyre blown out
            if fuel <= 0.0 or tyre_blowout[current_tyre.id]:
                is_limp = True

            if seg.seg_type == "straight":
                # Find next corner to determine required exit speed.
                # Wrap around to the start of segments (lap is a loop).
                seg_pos = segments.index(seg)
                next_corner: Optional[Segment] = None
                search_indices = list(range(seg_pos + 1, len(segments))) + list(range(0, seg_pos))
                for ns_idx in search_indices:
                    if segments[ns_idx].seg_type == "corner":
                        next_corner = segments[ns_idx]
                        break
                if next_corner is None:
                    # Pure straight-only track — no braking needed
                    exit_required = 999.0
                else:
                    exit_required = calc_max_corner_speed(
                        next_corner.radius, tyre_friction, car.crawl_constant
                    )

                ss = lap_strat.straight_strategies[straight_strat_idx]
                straight_strat_idx += 1

                seg_time, new_speed, seg_fuel, seg_deg = simulate_straight(
                    seg_length=seg.length,
                    entry_speed=current_speed,
                    target_speed=ss.target_speed,
                    brake_start=ss.brake_start,
                    exit_speed_required=exit_required,
                    accel=car.accel_m_s2 * weather.accel_multiplier,
                    decel=car.decel_m_s2 * weather.decel_multiplier,
                    deg_rate=eff_deg_rate,
                    weather_accel_mult=1.0,  # already folded in
                    weather_decel_mult=1.0,
                    is_limp=is_limp,
                    limp_speed=car.limp_constant,
                )

                elapsed += seg_time
                fuel -= seg_fuel
                total_fuel_used += seg_fuel
                fuel = max(fuel, 0.0)
                tyre_degradation[current_tyre.id] += seg_deg
                current_speed = new_speed

                if tyre_degradation[current_tyre.id] >= current_tyre.life_span:
                    tyre_blowout[current_tyre.id] = True
                    is_limp = True

            elif seg.seg_type == "corner":
                # Corner: check speed against max, apply crash if exceeded
                if is_limp:
                    corner_speed = car.limp_constant
                else:
                    corner_speed = current_speed

                max_speed = calc_max_corner_speed(
                    seg.radius, tyre_friction, car.crawl_constant
                )

                crashed = False
                if corner_speed > max_speed + 1e-9:
                    crashed = True
                    elapsed += car.crash_time_penalty
                    tyre_degradation[current_tyre.id] += car.crash_degradation
                    corner_speed = car.crawl_constant  # enter crawl mode

                # Time to traverse corner at corner_speed
                corner_time = seg.length / corner_speed if corner_speed > 0 else 1e12
                elapsed += corner_time

                # Fuel during corner
                seg_fuel = calc_fuel_consumption(corner_speed, corner_speed, seg.length)
                fuel -= seg_fuel
                total_fuel_used += seg_fuel
                fuel = max(fuel, 0.0)

                # Tyre degradation during corner
                seg_deg = calc_corner_degradation(corner_speed, seg.radius, eff_deg_rate)
                tyre_degradation[current_tyre.id] += seg_deg

                if tyre_degradation[current_tyre.id] >= current_tyre.life_span:
                    tyre_blowout[current_tyre.id] = True
                    is_limp = True

                # Exit speed: if crashed → crawl_constant, else corner_speed
                current_speed = car.crawl_constant if crashed else corner_speed

                if fuel <= 0.0:
                    is_limp = True

        # ── End of lap: process pit stop ─────────────────────────
        lap_pit = lap_strat
        if lap_pit.pit_enter:
            refuel = lap_pit.fuel_refuel_amount
            new_tyre_id = lap_pit.tyre_change_set_id

            pit_time = calc_pit_stop_time(
                refuel_amount=refuel if new_tyre_id is None else refuel,
                refuel_rate=car.refuel_rate,
                tyre_swap_time=car.tyre_swap_time if new_tyre_id is not None else 0.0,
                base_pit_stop_time=car.base_pit_stop_time,
            )
            elapsed += pit_time

            if refuel > 0:
                fuel = min(fuel + refuel, car.fuel_capacity)
                # NOTE: total_fuel_used tracks engine consumption only; refuelling does not reduce it.

            if new_tyre_id is not None:
                current_tyre = _find_tyre(config.tyre_sets, new_tyre_id)
                if new_tyre_id not in tyres_used:
                    tyres_used.append(new_tyre_id)
                is_limp = False  # fresh tyre clears blowout limp
                if fuel > 0:
                    is_limp = False  # fuel topped up

            current_speed = car.pit_exit_speed
        # End of lap

    # ── Build result ─────────────────────────────────────────────
    tyre_usage_list = []
    for tid in tyres_used:
        tyre_usage_list.append(TyreUsage(
            tyre_set_id=tid,
            total_degradation=tyre_degradation[tid],
            blew_out=tyre_blowout[tid],
        ))

    num_blowouts = sum(1 for u in tyre_usage_list if u.blew_out)

    return SimResult(
        total_time=elapsed,
        fuel_used=max(total_fuel_used, 0.0),
        tyre_usages=tyre_usage_list,
        num_blowouts=num_blowouts,
        valid=True,
    )
