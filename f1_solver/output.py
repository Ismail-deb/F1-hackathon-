"""Convert a Strategy object to the required output JSON format."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .models import RaceConfig, Segment, Strategy


def build_output(config: RaceConfig, strategy: Strategy) -> Dict[str, Any]:
    """Build the output dictionary matching the required schema."""
    segments: List[Segment] = config.track.segments
    straights = [s for s in segments if s.seg_type == "straight"]

    out_laps = []
    for lap_idx, lap_strat in enumerate(strategy.laps):
        out_segments: List[Dict[str, Any]] = []
        straight_strat_iter = iter(lap_strat.straight_strategies)

        for seg in segments:
            if seg.seg_type == "straight":
                ss = next(straight_strat_iter)
                out_segments.append({
                    "id": seg.id,
                    "type": "straight",
                    "target_m/s": round(ss.target_speed, 4),
                    "brake_start_m_before_next": round(ss.brake_start, 4),
                })
            else:  # corner
                out_segments.append({
                    "id": seg.id,
                    "type": "corner",
                })

        # Pit block
        pit_block: Dict[str, Any] = {"enter": lap_strat.pit_enter}
        if lap_strat.pit_enter:
            if lap_strat.tyre_change_set_id is not None:
                pit_block["tyre_change_set_id"] = lap_strat.tyre_change_set_id
            if lap_strat.fuel_refuel_amount > 0.0:
                pit_block["fuel_refuel_amount_l"] = round(lap_strat.fuel_refuel_amount, 4)

        out_laps.append({
            "lap": lap_idx + 1,
            "segments": out_segments,
            "pit": pit_block,
        })

    return {
        "initial_tyre_id": strategy.initial_tyre_id,
        "laps": out_laps,
    }


def write_output(config: RaceConfig, strategy: Strategy, path: str) -> None:
    """Serialize the strategy to a JSON file."""
    data = build_output(config, strategy)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
