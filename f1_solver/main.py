#!/usr/bin/env python3
"""
F1 Race Simulation Solver — CLI entry point.

Usage:
    python -m f1_solver.main <input.json> <output.json> [--level N] [--time-budget S]

Options:
    --level N         Force competition level (1-4). Auto-detected if omitted.
    --time-budget S   Max optimization time in seconds (default: 60).
    --naive           Skip optimization, use a naive strategy (for quick testing).
"""
from __future__ import annotations

import argparse
import sys
import time

from .optimizer import optimize, _naive_strategy
from .output import write_output
from .parser import parse_input
from .scorer import calc_score
from .simulator import simulate_race


def main() -> None:
    parser = argparse.ArgumentParser(description="F1 Race Strategy Solver")
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("output", help="Path to output JSON file")
    parser.add_argument("--level", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Competition level (auto-detected if omitted)")
    parser.add_argument("--time-budget", type=float, default=60.0,
                        help="Max optimization time in seconds (default: 60)")
    parser.add_argument("--naive", action="store_true",
                        help="Use a naive strategy (skips optimizer, useful for testing)")
    args = parser.parse_args()

    # ── Parse input ───────────────────────────────────────────────
    print(f"[info] Reading input: {args.input}", file=sys.stderr)
    config = parse_input(args.input)

    if args.level is not None:
        config.level = args.level

    print(f"[info] Level: {config.level}", file=sys.stderr)
    print(f"[info] Track: {config.track.total_laps} laps, "
          f"{len(config.track.segments)} segments "
          f"({config.track.num_straights} straights)", file=sys.stderr)
    print(f"[info] Tyre sets: {len(config.tyre_sets)}", file=sys.stderr)

    # ── Optimize or use naive ─────────────────────────────────────
    t0 = time.time()
    if args.naive:
        print("[info] Using naive strategy (no optimization)", file=sys.stderr)
        strategy = _naive_strategy(config)
    else:
        print(f"[info] Optimizing (budget: {args.time_budget}s) ...", file=sys.stderr)
        strategy = optimize(config, time_budget_s=args.time_budget)

    elapsed = time.time() - t0
    print(f"[info] Optimization done in {elapsed:.1f}s", file=sys.stderr)

    # ── Simulate final strategy for reporting ─────────────────────
    result = simulate_race(config, strategy)
    score = calc_score(config, result)

    print(f"[result] Total time : {result.total_time:.3f}s", file=sys.stderr)
    print(f"[result] Fuel used  : {result.fuel_used:.2f}L", file=sys.stderr)
    print(f"[result] Blowouts   : {result.num_blowouts}", file=sys.stderr)
    print(f"[result] Score      : {score:.2f}", file=sys.stderr)

    # ── Write output ──────────────────────────────────────────────
    write_output(config, strategy, args.output)
    print(f"[info] Output written: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
