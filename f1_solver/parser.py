"""Parse input JSON into RaceConfig model objects."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .constants import BASE_FRICTION
from .models import (
    Car,
    RaceConfig,
    Segment,
    Track,
    TyreSet,
    WeatherCondition,
)


def parse_input(path: str) -> RaceConfig:
    with open(path, "r") as f:
        data = json.load(f)
    return _build_config(data)


def parse_input_from_dict(data: Dict[str, Any]) -> RaceConfig:
    return _build_config(data)


def _build_config(data: Dict[str, Any]) -> RaceConfig:
    # ── Car ───────────────────────────────────────────────────────
    cd = data["car"]
    car = Car(
        accel_m_s2=cd["accel_m/s2"],
        decel_m_s2=cd["decel_m/s2"],
        base_pit_stop_time=cd.get("base_pit_stop_time_s", 0.0),
        tyre_swap_time=cd.get("tyre_swap_time_s", 0.0),
        refuel_rate=cd.get("refuel_rate_l/s", 1.0),
        pit_exit_speed=cd.get("pit_exit_speed_m/s", 0.0),
        crawl_constant=cd.get("crawl_constant_m/s", 5.0),
        limp_constant=cd.get("limp_constant_m/s", 5.0),
        fuel_capacity=cd.get("fuel_capacity_l", 1e9),
        start_speed=cd.get("start_speed_m/s", 0.0),
        crash_time_penalty=cd.get("crash_time_penalty_s", 0.0),
        crash_degradation=cd.get("crash_degradation", 0.1),
    )

    # ── Track ─────────────────────────────────────────────────────
    segments = _parse_segments(data["track"]["segments"])
    track = Track(
        segments=segments,
        total_laps=data["track"]["total_laps"],
    )

    # ── Tyre Sets ─────────────────────────────────────────────────
    tyre_sets = _parse_tyres(data.get("tyre_sets", []))

    # ── Weather ───────────────────────────────────────────────────
    weather = _parse_weather(data.get("weather", []))

    # ── Level detection ───────────────────────────────────────────
    level = data.get("level", _detect_level(data))

    # ── Scoring params ────────────────────────────────────────────
    scoring = data.get("scoring", {})
    time_reference = scoring.get("time_reference_s", data.get("time_reference_s", 0.0))
    fuel_soft_cap = scoring.get("fuel_soft_cap_l", data.get("fuel_soft_cap_l", 0.0))

    initial_fuel = data.get("initial_fuel_l", car.fuel_capacity)

    return RaceConfig(
        car=car,
        track=track,
        tyre_sets=tyre_sets,
        weather=weather,
        level=level,
        time_reference=time_reference,
        fuel_soft_cap=fuel_soft_cap,
        initial_fuel=initial_fuel,
    )


def _parse_segments(seg_list: List[Dict]) -> List[Segment]:
    segments = []
    for s in seg_list:
        segments.append(Segment(
            id=s["id"],
            seg_type=s["type"],
            length=s["length_m"],
            radius=s.get("radius_m", 0.0),
        ))
    return segments


def _parse_tyres(tyre_list: List[Dict]) -> List[TyreSet]:
    tyres = []
    for t in tyre_list:
        compound = t["compound"].lower()
        tyres.append(TyreSet(
            id=t["id"],
            compound=compound,
            base_friction=BASE_FRICTION.get(compound, 1.5),
            degradation_rate=t.get("degradation_rate", 1.0),
            life_span=t.get("life_span", 1e9),
        ))
    return tyres


def _parse_weather(weather_list: List[Dict]) -> List[WeatherCondition]:
    conditions = []
    for w in weather_list:
        conditions.append(WeatherCondition(
            start_time=w.get("start_time_s", 0.0),
            accel_multiplier=w.get("accel_multiplier", 1.0),
            decel_multiplier=w.get("decel_multiplier", 1.0),
            friction_multiplier=w.get("friction_multiplier", 1.0),
            degradation_rate_multiplier=w.get("degradation_rate_multiplier", 1.0),
        ))
    # Sort by start_time ascending
    conditions.sort(key=lambda c: c.start_time)
    return conditions


def _detect_level(data: Dict) -> int:
    has_weather = bool(data.get("weather"))
    has_fuel_cap = "fuel_soft_cap_l" in data or "fuel_soft_cap_l" in data.get("scoring", {})
    has_degradation = any(
        t.get("degradation_rate", 0) > 0 and t.get("life_span", 1e9) < 1e8
        for t in data.get("tyre_sets", [])
    )
    if has_degradation:
        return 4
    if has_weather:
        return 3
    if has_fuel_cap:
        return 2
    return 1
