"""Recompute the four requested outputs and compare the paper tables.

Run from the repository root: python scripts/recompute_post_contest.py
Historical competition outputs remain untouched. Full workbook replicas are
written to _tmp/post_contest/reproduction; durable evidence goes to reports/.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from drymodel import config, production


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cfg = config.load_config()
    target = ROOT / "_tmp" / "post_contest" / "reproduction"
    report_dir = ROOT / "reports" / "post_contest"
    report_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    sources_before = {str(p.relative_to(ROOT)): sha(p)
                      for p in sorted((ROOT / "src" / "drymodel").glob("*.py"))}
    config_before = sha(cfg.source_path)
    original = {str(p.relative_to(ROOT)): sha(p)
                for p in (ROOT / "outputs").glob("*") if p.is_file()}
    print("Recomputing Q1-Q4 at the configured production resolution...", flush=True)
    result = production.run_production(cfg, outputs_dir=target)
    evidence = {
        "scope": "Full result1-4 generation and V8/V9; all paper table cells compared with historical outputs",
        "source_sha256": sources_before,
        "source_files_unchanged": sources_before == {
            str(p.relative_to(ROOT)): sha(p)
            for p in sorted((ROOT / "src" / "drymodel").glob("*.py"))},
        "config_sha256": config_before,
        "config_file_unchanged": config_before == sha(cfg.source_path),
        "historical_outputs_sha256": original,
        "production_ok": bool(result["ok"]),
        "table_comparisons": {},
    }
    if result["ok"]:
        tolerance = cfg.raw["acceptance"]["table_points"]
        for path in sorted((ROOT / "outputs").glob("table*.csv")):
            old = np.loadtxt(path, delimiter=",", skiprows=1)
            new = np.loadtxt(target / path.name, delimiter=",", skiprows=1)
            if old.shape != new.shape or not np.isfinite(new).all():
                raise RuntimeError(f"Invalid table shape or values: {path.name}")
            delta = np.abs(new-old)
            # Printed times may differ by one rounding unit after re-integration.
            time_tol = 1e-4 if "t_h" in path.read_text(encoding="utf-8").splitlines()[0] else 0.0
            value_tol = float(tolerance["dT_degC"] if "temp" in path.name else tolerance["dC"])
            # Radius is prescribed, not a fitted field; compare displayed values.
            if "radius" in path.name:
                value_tol = 1e-4
            ok = bool(np.all(delta[:, 0] <= time_tol+1e-12)
                      and np.all(delta[:, 1:] <= value_tol+1e-12))
            evidence["table_comparisons"][path.name] = {
                "max_abs_value_difference": float(delta[:, 1:].max()),
                "max_time_difference": float(delta[:, 0].max()),
                "value_tolerance": value_tol,
                "time_tolerance": time_tol,
                "changed_displayed_cells": int(np.count_nonzero(delta)),
                "pass": ok,
                "recomputed_rows": new.tolist(),
            }
        historical = json.loads((ROOT / "outputs" / "production_receipt_supplementary.json").read_text(encoding="utf-8"))
        events = {}
        for q in ("q23", "q4"):
            got = result[q]
            previous = historical["results"][q]
            diff = abs(got["t_star_s"] - previous["t_star_s"])/3600
            events[q] = {k: got[k] for k in ("t_star_s", "t_star_h", "t_sample_s", "post_max_cmax", "cmax_at_tsample")}
            events[q].update({"difference_from_historical_h": diff,
                              "tolerance_h": float(cfg.raw["acceptance"]["t_star_h"]),
                              "pass": bool(diff <= cfg.raw["acceptance"]["t_star_h"] and got["t_sample_s"] == previous["t_sample_s"])})
        evidence["events"] = events
        evidence["V8"] = result["V8"]
        evidence["V9"] = result["V9"]
        evidence["effective_config"] = result["fc"]
        evidence["all_ok"] = (all(x["pass"] for x in events.values()) and
                              all(x["pass"] for x in evidence["table_comparisons"].values()))
    else:
        evidence["all_ok"] = False
        evidence["failure"] = result.get("reason", "unknown")
    evidence["historical_outputs_unchanged"] = all(sha(ROOT/p) == digest for p, digest in original.items())
    evidence["all_ok"] = bool(evidence["all_ok"] and evidence["historical_outputs_unchanged"]
                              and evidence["source_files_unchanged"] and evidence["config_file_unchanged"])
    evidence["elapsed_seconds"] = perf_counter()-started
    evidence["production_receipt"] = json.loads((target / "production_receipt.json").read_text(encoding="utf-8"))
    (report_dir / "reproduction.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"all_ok": evidence["all_ok"], "elapsed_seconds": evidence["elapsed_seconds"], "events": evidence.get("events")}, ensure_ascii=False), flush=True)
    return 0 if evidence["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
