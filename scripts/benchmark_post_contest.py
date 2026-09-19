"""Reproduce round-2 numerical performance decisions without changing equations.

Run from the repository root: python scripts/benchmark_post_contest.py
Old implementations are retained below so the timing/accuracy comparison remains
reproducible after the production implementation has been replaced. This script
patches a function only within its own process and restores it after every run.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drymodel import config, props as P, runners
from drymodel.solver_bdf import integrate_bdf


def integral_before(Ci, Cj, Tb, Dfun, npts=8):
    """Exact implementation at the round-1 commit (4e066e0)."""
    Ci = np.asarray(Ci, dtype=np.float64)
    Cj = np.asarray(Cj, dtype=np.float64)
    Tb = np.asarray(Tb, dtype=np.float64)
    xi, wt = P._gauss01(npts)
    acc = np.zeros(np.broadcast(Ci, Cj, Tb).shape, dtype=np.float64)
    for xk, wk in zip(xi, wt):
        Cmid = (1.0 - xk) * Ci + xk * Cj
        acc = acc + wk * Dfun(Cmid, Tb)
    return acc


def dense_before(result, times):
    """Production scalar dense evaluation, with all round-1 guards retained."""
    if not result.ok or not result.segments:
        raise ValueError(f"No successful BDF trajectory: {result.message}")
    t = np.atleast_1d(np.asarray(times, dtype=np.float64))
    if t.ndim != 1 or not np.all(np.isfinite(t)):
        raise ValueError("Expected finite scalar or 1D sample times")
    out = np.empty((t.size, result.segments[0].y.shape[0]))
    for k, ti in enumerate(t):
        out[k] = result._find_segment(ti).sol(ti)
    return out


def dense_batched_candidate(result, times, max_values=131072):
    """Rejected candidate: sorted per-segment evaluation in bounded blocks.

    Retained solely to reproduce the measured rejection; not used by solvers.
    The first covering segment wins at shared endpoints, as in dense_before.
    """
    if not result.ok or not result.segments:
        raise ValueError(f"No successful BDF trajectory: {result.message}")
    t = np.atleast_1d(np.asarray(times, dtype=np.float64))
    if t.ndim != 1 or not np.all(np.isfinite(t)):
        raise ValueError("Expected finite scalar or 1D sample times")
    out = np.empty((t.size, result.segments[0].y.shape[0]))
    pending = np.ones(t.size, dtype=bool)
    block_size = max(1, max_values // out.shape[1])
    for seg in result.segments:
        index = np.flatnonzero(pending & (seg.t[0] <= t) & (t <= seg.t[-1]))
        index = index[np.argsort(t[index], kind="stable")]
        for start in range(0, index.size, block_size):
            rows = index[start:start + block_size]
            out[rows] = seg.sol(t[rows]).T
        pending[index] = False
    if np.any(pending):
        result._find_segment(t[np.flatnonzero(pending)[0]])
    return out


@contextmanager
def use_integral(implementation):
    previous = P.D_face_integral
    P.D_face_integral = implementation
    try:
        yield
    finally:
        P.D_face_integral = previous


def paired_timing(before, after, *, repeat=9, loops=100):
    """Alternate order to reduce timing drift; do not assert wall-clock speed."""
    before()
    after()
    samples = {"before": [], "after": []}
    for iteration in range(repeat):
        order = (("before", before), ("after", after))
        if iteration % 2:
            order = order[::-1]
        for label, fn in order:
            start = perf_counter()
            for _ in range(loops):
                fn()
            samples[label].append((perf_counter() - start) / loops)
    medians = {key: float(np.median(value)) for key, value in samples.items()}
    return {"repeat": repeat, "calls_per_repeat": loops,
            "seconds_per_call": samples, "median_seconds": medians,
            "speedup_median_before_over_after": medians["before"] / medians["after"]}


def errors(before, after):
    before, after = np.asarray(before), np.asarray(after)
    delta = np.abs(before - after)
    nonzero = before != 0
    return {"max_absolute": float(np.max(delta, initial=0)),
            "max_relative_nonzero": float(np.max(delta[nonzero] / np.abs(before[nonzero]), initial=0)),
            "bitwise_equal": bool(np.array_equal(before, after))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/post_contest/performance.json")
    parser.add_argument("--N", type=int, default=800)
    parser.add_argument("--repeat", type=int, default=9)
    args = parser.parse_args()
    if args.N < 20 or args.repeat < 3:
        parser.error("N >= 20 and repeat >= 3 are required")
    current = P.D_face_integral
    cfg = config.load_config()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": "4e066e0",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "platform": platform.platform()},
        "scope": "Same appendix formulas, quadrature nodes, tolerances, boundary conditions and mesh.",
        "interpretation": "Timing applies to this machine/run, not a universal performance guarantee.",
        "N": args.N, "integral": [], "rhs": [], "trajectory": {}, "dense_output_candidate": [],
    }
    for question in ("q1", "q23", "q4"):
        prop = cfg.props(question)
        for npts in (8, 16):
            c = np.linspace(.05, cfg.C0, args.N)
            temp = np.linspace(cfg.T0_K, 323.15, args.N)
            call_args = (c, c * 1.002, temp, prop.D, npts)
            before = lambda: integral_before(*call_args)
            after = lambda: current(*call_args)
            report["integral"].append({"question": question, "npts": npts,
                "error": errors(before(), after()),
                "timing": paired_timing(before, after, repeat=args.repeat)})

    op, _ = runners.build_fixed_operator(cfg, "q23", args.N, interface="integral")
    y0 = runners.initial_state(cfg, args.N)
    trajectory_end = 1800.0
    def solve_with(implementation):
        with use_integral(implementation):
            result = integrate_bdf(op, y0, 0., trajectory_end, cfg.bdf, breakpoints=(900.,))
        if not result.ok:
            raise RuntimeError(result.message)
        return result
    old_trajectory, new_trajectory = solve_with(integral_before), solve_with(current)
    sample_times = np.linspace(0., trajectory_end, 301)
    old_y, new_y = old_trajectory.eval(sample_times), new_trajectory.eval(sample_times)
    report["trajectory"] = {
        "question": "q23", "npts": 8, "t_end_s": trajectory_end,
        "breakpoints_s": [900.], "bdf_controls": cfg.bdf,
        "C_error": errors(old_y[:, :args.N+1], new_y[:, :args.N+1]),
        "T_error_K": errors(old_y[:, args.N+1:], new_y[:, args.N+1:]),
        "old_steps": [len(seg.t)-1 for seg in old_trajectory.segments],
        "new_steps": [len(seg.t)-1 for seg in new_trajectory.segments],
        "timing": paired_timing(lambda: solve_with(integral_before), lambda: solve_with(current),
                                repeat=args.repeat, loops=1),
    }
    # RHS timing uses an evolved physical profile, not only the uniform initial state.
    profile = new_trajectory.eval(900.)[0]
    for npts in (8, 16):
        op.integral_npts = npts
        def rhs_with(implementation):
            with use_integral(implementation):
                return op.rhs(900., profile)
        report["rhs"].append({"question": "q23", "npts": npts, "profile_time_s": 900.,
            "error": errors(rhs_with(integral_before), rhs_with(current)),
            "timing": paired_timing(lambda: rhs_with(integral_before), lambda: rhs_with(current),
                                    repeat=args.repeat)})

    times = np.r_[np.linspace(0., trajectory_end, 3001), 900., 0., trajectory_end, 900.]
    for ordering in ("sorted_with_duplicates", "shuffled_with_duplicates"):
        if ordering.startswith("sorted"):
            times.sort()
        else:
            np.random.default_rng(20260918).shuffle(times)
        expected = dense_before(new_trajectory, times)
        for max_values in (12816, 32768, 131072, 524288):
            before = lambda: dense_before(new_trajectory, times)
            after = lambda: dense_batched_candidate(new_trajectory, times, max_values)
            report["dense_output_candidate"].append({
                "ordering": ordering, "sample_count": len(times), "max_values_per_block": max_values,
                "error": errors(expected, after()),
                "timing": paired_timing(before, after, repeat=args.repeat, loops=1)})
    report["decisions"] = {
        "integral_vectorization": "adopted: substantial repeatable RHS speedup; unchanged equations and quadrature",
        "dense_batching": "rejected: speedup is inconsistent and larger blocks regress on the N800 workload; scalar production eval retained",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "report": str(args.output),
        "integral_speedups": [{"q": row["question"], "npts": row["npts"],
                               "speedup": row["timing"]["speedup_median_before_over_after"]}
                              for row in report["integral"]],
        "rhs_speedups": [row["timing"]["speedup_median_before_over_after"] for row in report["rhs"]],
        "trajectory_speedup": report["trajectory"]["timing"]["speedup_median_before_over_after"],
        "trajectory_C_error": report["trajectory"]["C_error"],
        "trajectory_T_error": report["trajectory"]["T_error_K"],
        "dense_speedups": [row["timing"]["speedup_median_before_over_after"] for row in report["dense_output_candidate"]],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
