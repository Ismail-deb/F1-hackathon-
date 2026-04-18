"""Pure physics calculation functions.

All functions are stateless and side-effect-free so they can be called
freely from the simulator and from unit tests.
"""
from __future__ import annotations

import math
from typing import Tuple

from .constants import G


# ── Tyre friction ────────────────────────────────────────────────────

def calc_tyre_friction(
    base_friction: float,
    total_degradation: float,
    weather_friction_mult: float,
) -> float:
    """Effective tyre friction coefficient."""
    return (base_friction - total_degradation) * weather_friction_mult


def calc_max_corner_speed(
    radius: float,
    tyre_friction: float,
    crawl_constant: float,
) -> float:
    """Maximum safe speed through a corner (m/s)."""
    return math.sqrt(max(tyre_friction * G * radius, 0.0)) + crawl_constant


# ── Acceleration / deceleration kinematics ──────────────────────────

def calc_accel_distance(v_initial: float, v_final: float, accel: float) -> float:
    """Distance (m) to change speed from v_initial to v_final under constant accel."""
    return (v_final ** 2 - v_initial ** 2) / (2.0 * accel)


def calc_accel_time(v_initial: float, v_final: float, accel: float) -> float:
    """Time (s) to change speed from v_initial to v_final under constant accel."""
    return (v_final - v_initial) / accel


def calc_cruise_time(distance: float, speed: float) -> float:
    """Time (s) to travel distance at constant speed."""
    return distance / speed if speed > 0 else 1e12


# ── Fuel consumption ────────────────────────────────────────────────

def calc_fuel_consumption(v_initial: float, v_final: float, distance: float) -> float:
    """Fuel consumed (litres) over a distance with given entry/exit speeds."""
    v_avg = (v_initial + v_final) / 2.0
    return (0.0005 + 0.0000000015 * v_avg ** 2) * distance


# ── Tyre degradation ────────────────────────────────────────────────

def calc_straight_degradation(rate: float, length: float) -> float:
    """Tyre degradation from rolling along a straight."""
    return rate * length * 0.0000166


def calc_brake_degradation(v_initial: float, v_final: float, rate: float) -> float:
    """Tyre degradation from braking (v_initial > v_final)."""
    return ((v_initial / 100.0) ** 2 - (v_final / 100.0) ** 2) * 0.0398 * rate


def calc_corner_degradation(speed: float, radius: float, rate: float) -> float:
    """Tyre degradation from cornering."""
    return 0.000265 * (speed ** 2 / radius) * rate


# ── Pit stop time ───────────────────────────────────────────────────

def calc_pit_stop_time(
    refuel_amount: float,
    refuel_rate: float,
    tyre_swap_time: float,
    base_pit_stop_time: float,
) -> float:
    """Total time spent stationary in pit lane (s)."""
    fuel_time = refuel_amount / refuel_rate if refuel_rate > 0 else 0.0
    return fuel_time + tyre_swap_time + base_pit_stop_time


# ── Straight segment simulation ──────────────────────────────────────

def simulate_straight(
    seg_length: float,
    entry_speed: float,
    target_speed: float,
    brake_start: float,
    exit_speed_required: float,  # max speed allowed at end (corner max speed)
    accel: float,
    decel: float,
    deg_rate: float,
    weather_accel_mult: float = 1.0,
    weather_decel_mult: float = 1.0,
    is_limp: bool = False,
    limp_speed: float = 5.0,
) -> Tuple[float, float, float, float]:
    """
    Simulate one straight segment.

    Returns (time_s, exit_speed_m_s, fuel_l, degradation).
    
    Physics:
    - Phase 1: accelerate from entry_speed up to target_speed
    - Phase 2: cruise at target_speed
    - Phase 3: brake from target_speed down to exit_speed_required

    Edge cases handled:
    - Limp mode: constant limp_speed throughout
    - entry_speed > target_speed: no acceleration phase, car holds entry_speed
    - Braking distance clamp: if brake_start is too large, cap it at seg_length
    - Insufficient distance to reach target_speed: target is capped at achievable speed
    - Exit speed already satisfies corner: no additional braking needed
    """
    if is_limp:
        t = seg_length / limp_speed
        fuel = calc_fuel_consumption(limp_speed, limp_speed, seg_length)
        return t, limp_speed, fuel, 0.0

    eff_accel = accel * weather_accel_mult
    eff_decel = decel * weather_decel_mult

    # Effective target: never lower than entry (car can't decelerate mid-straight)
    eff_target = max(target_speed, entry_speed)

    # Required braking distance to reach exit_speed from eff_target
    exit_capped = min(exit_speed_required, eff_target)
    min_brake_dist = calc_accel_distance(exit_capped, eff_target, eff_decel) if eff_target > exit_capped else 0.0

    # Clamp brake_start to [min_brake_dist, seg_length]
    brake_dist_used = max(min_brake_dist, min(brake_start, seg_length))

    # Distance available before braking zone
    pre_brake_length = seg_length - brake_dist_used

    # Acceleration phase: from entry_speed toward eff_target
    accel_dist = calc_accel_distance(entry_speed, eff_target, eff_accel) if eff_target > entry_speed else 0.0

    if accel_dist > pre_brake_length:
        # Not enough room to reach eff_target — find peak speed
        # peak speed v satisfies: (v²-v_e²)/(2a) + (v²-v_x²)/(2d) = seg_length
        # (1/(2a) + 1/(2d)) * v² = seg_length + v_e²/(2a) + v_x²/(2d)
        a_inv = 1.0 / (2.0 * eff_accel)
        d_inv = 1.0 / (2.0 * eff_decel) if eff_decel > 0 else 0.0
        num = seg_length + entry_speed ** 2 * a_inv + exit_capped ** 2 * d_inv
        denom = a_inv + d_inv
        peak_v_sq = num / denom if denom > 0 else entry_speed ** 2
        peak_v = math.sqrt(max(peak_v_sq, entry_speed ** 2))
        peak_v = min(peak_v, eff_target)

        # Recalculate distances with peak_v
        a_dist = calc_accel_distance(entry_speed, peak_v, eff_accel) if peak_v > entry_speed else 0.0
        b_dist = calc_accel_distance(exit_capped, peak_v, eff_decel) if peak_v > exit_capped else 0.0
        c_dist = max(seg_length - a_dist - b_dist, 0.0)

        t_a = calc_accel_time(entry_speed, peak_v, eff_accel) if peak_v > entry_speed else 0.0
        t_c = calc_cruise_time(c_dist, peak_v) if c_dist > 0 else 0.0
        t_b = calc_accel_time(exit_capped, peak_v, eff_decel) if peak_v > exit_capped else 0.0

        total_dist = a_dist + c_dist + b_dist
        fuel = calc_fuel_consumption(entry_speed, peak_v, a_dist) + \
               calc_fuel_consumption(peak_v, peak_v, c_dist) + \
               calc_fuel_consumption(peak_v, exit_capped, b_dist)

        # Degradation
        deg = calc_straight_degradation(deg_rate, a_dist + c_dist)
        deg += calc_brake_degradation(peak_v, exit_capped, deg_rate) if peak_v > exit_capped else 0.0

        actual_exit = exit_capped
        return t_a + t_c + t_b, actual_exit, fuel, deg

    # Normal case: enough room
    t_a = calc_accel_time(entry_speed, eff_target, eff_accel) if eff_target > entry_speed else 0.0
    cruise_dist = pre_brake_length - (accel_dist if eff_target > entry_speed else 0.0)
    t_c = calc_cruise_time(cruise_dist, eff_target) if cruise_dist > 0 else 0.0

    actual_exit = exit_capped
    t_b = calc_accel_time(actual_exit, eff_target, eff_decel) if eff_target > actual_exit else 0.0

    a_dist_used = accel_dist if eff_target > entry_speed else 0.0
    b_dist_used = calc_accel_distance(actual_exit, eff_target, eff_decel) if eff_target > actual_exit else 0.0

    fuel = calc_fuel_consumption(entry_speed, eff_target, a_dist_used) + \
           calc_fuel_consumption(eff_target, eff_target, cruise_dist) + \
           calc_fuel_consumption(eff_target, actual_exit, b_dist_used)

    deg = calc_straight_degradation(deg_rate, a_dist_used + cruise_dist)
    deg += calc_brake_degradation(eff_target, actual_exit, deg_rate) if eff_target > actual_exit else 0.0

    return t_a + t_c + t_b, actual_exit, fuel, deg
