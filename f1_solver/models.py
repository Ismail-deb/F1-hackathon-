from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


# ── Track / Input Models ─────────────────────────────────────────────

@dataclass
class Segment:
    id: int
    seg_type: str          # "straight" or "corner"
    length: float          # metres
    radius: float = 0.0   # only meaningful for corners


@dataclass
class Track:
    segments: List[Segment]
    total_laps: int
    lap_length: float = 0.0  # sum of segment lengths (computed)

    def __post_init__(self):
        self.lap_length = sum(s.length for s in self.segments)

    @property
    def straights(self) -> List[Segment]:
        return [s for s in self.segments if s.seg_type == "straight"]

    @property
    def corners(self) -> List[Segment]:
        return [s for s in self.segments if s.seg_type == "corner"]

    @property
    def num_straights(self) -> int:
        return len(self.straights)


@dataclass
class Car:
    accel_m_s2: float
    decel_m_s2: float
    base_pit_stop_time: float
    tyre_swap_time: float
    refuel_rate: float           # litres per second
    pit_exit_speed: float
    crawl_constant: float
    limp_constant: float
    fuel_capacity: float = 1e9   # effectively infinite for Level 1
    start_speed: float = 0.0
    crash_time_penalty: float = 0.0
    crash_degradation: float = 0.1


@dataclass
class TyreSet:
    id: int
    compound: str            # "soft", "medium", "hard", "intermediate", "wet"
    base_friction: float
    degradation_rate: float
    life_span: float         # blowout threshold


@dataclass
class WeatherCondition:
    start_time: float        # race-elapsed seconds when this condition starts
    accel_multiplier: float
    decel_multiplier: float
    friction_multiplier: float
    degradation_rate_multiplier: float


@dataclass
class RaceConfig:
    car: Car
    track: Track
    tyre_sets: List[TyreSet]
    weather: List[WeatherCondition]
    level: int
    time_reference: float = 0.0
    fuel_soft_cap: float = 0.0
    initial_fuel: float = 0.0


# ── Strategy Models (what the solver decides) ────────────────────────

@dataclass
class StraightStrategy:
    target_speed: float          # m/s
    brake_start: float           # metres before end of straight


@dataclass
class LapStrategy:
    straight_strategies: List[StraightStrategy]
    pit_enter: bool = False
    tyre_change_set_id: Optional[int] = None
    fuel_refuel_amount: float = 0.0


@dataclass
class Strategy:
    initial_tyre_id: int
    laps: List[LapStrategy]


# ── Simulation Result ────────────────────────────────────────────────

@dataclass
class TyreUsage:
    tyre_set_id: int
    total_degradation: float = 0.0
    blew_out: bool = False


@dataclass
class SimResult:
    total_time: float = 0.0
    fuel_used: float = 0.0
    tyre_usages: List[TyreUsage] = field(default_factory=list)
    num_blowouts: int = 0
    valid: bool = True
