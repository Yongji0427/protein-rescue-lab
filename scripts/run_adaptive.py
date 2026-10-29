"""Matched direct-feedback versus contrast-feedback development experiments."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from adaptive_policy import AdaptivePolicy
from run_pilot import QueryOracle, count_interval


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "gb1_adaptive_v1"
PILOT = ROOT / "experiments" / "gb1_pilot_v1"


def main():
    start = time.perf_counter()
    protocol_file = OUT / "protocol.json"
    protocol = json.loads(protocol_file.read_text())
    tasks = pd.read_csv(PILOT / "development_tasks.csv")
    singles = pd.read_csv(ROOT / "data/public_reference/singles.csv").set_index("mutant")
    raw = pd.read_csv(ROOT / "data/evaluator_only/gb1_raw.csv")
    held = protocol["holdout_full_sequence_positions"]
    raw = raw.loc[raw["Mut2 Position"].notna() & ~(raw["Mut1 Position"] + 226).isin(held) & ~(raw["Mut2 Position"] + 226).isin(held)].copy()
    parts = raw.mutant.str.split(":", expand=True)
    raw["r1"], raw["r2"] = parts[0], parts[1]
    raw["ci_low"], _ = count_interval(raw)
    raw["W_display"] = np.exp(raw.lnW)
    groups1 = {key: group for key, group in raw.groupby("r1", sort=False)}
    groups2 = {key: group for key, group in raw.groupby("r2", sort=False)}
    empty = raw.iloc[0:0]
    records = []
    fields = ["damage", "method", "repeat", "query", "batch_cost", "rescue", "slot", "lnW", "W_display", "count_ci_low", "wt_rescue_lnW", "predicted_lnW_before_batch"]
    with gzip.open(OUT / "query_log.csv.gz", "wt", newline="") as log:
        writer = csv.DictWriter(log, fieldnames=fields)
        writer.writeheader()
        for task in tasks.itertuples():
            frame = pd.concat([groups1.get(task.damage, empty), groups2.get(task.damage, empty)]).copy()
            frame["rescue"] = np.where(frame.r1 == task.damage, frame.r2, frame.r1)
            frame = frame.sort_values("rescue").reset_index(drop=True)
            assert len(frame) == task.candidates
            ids = frame.rescue.to_numpy()
            wt = frame.rescue.map(singles.lnW).to_numpy()
            damage_y = float(singles.loc[task.damage, "lnW"])
            labels = {row.rescue: row for row in frame.itertuples()}
            for repeat in range(protocol["repeats"]):
                seed = np.random.SeedSequence([protocol["seed"], task.full_position, ord(task.damage[-1]), repeat])
                for method in protocol["methods"]:
                    policy = AdaptivePolicy(ids, wt, damage_y, method, seed)
                    oracle = QueryOracle(task.damage, labels, 32)
                    best = float(np.exp(damage_y))
                    robust_best, first, first_robust = 0.0, None, None
                    for cost in range(4, 33, 4):
                        predicted = policy.predict()
                        batch = policy.choose_batch()
                        indices, feedback = [], []
                        # All four choices are committed before any of their labels is revealed.
                        for idx, slot in batch:
                            row = oracle.query(str(ids[idx]))
                            indices.append(idx)
                            feedback.append(row.lnW)
                            best = max(best, row.W_display)
                            robust_best = max(robust_best, row.ci_low)
                            if first is None and row.W_display >= .5:
                                first = cost
                            if first_robust is None and row.ci_low >= .5:
                                first_robust = cost
                            writer.writerow(dict(zip(fields, [task.damage, method, repeat, oracle.used, cost, ids[idx], slot, row.lnW, row.W_display, row.ci_low, wt[idx], predicted[idx]])))
                        policy.update(indices, feedback)
                        assert oracle.used == cost
                        records.append({"damage": task.damage, "full_position": task.full_position, "method": method,
                            "repeat": repeat, "budget": cost, "best_W": best, "success": best >= .5,
                            "count_robust_success": robust_best >= .5, "strong_success": best >= 1,
                            "first_success_batch_cost": first, "first_robust_success_batch_cost": first_robust,
                            "queries_used": oracle.used})
    runs = pd.DataFrame(records)
    runs.to_csv(OUT / "runs.csv", index=False)
    summary = runs.groupby(["method", "budget"]).agg(success_rate=("success", "mean"),
        count_robust_success_rate=("count_robust_success", "mean"), strong_success_rate=("strong_success", "mean"),
        mean_best_W=("best_W", "mean"), median_best_W=("best_W", "median")).reset_index()
    summary.to_csv(OUT / "summary.csv", index=False)
    per_task = runs.groupby(["damage", "method", "budget"]).agg(success_rate=("success", "mean"),
        count_robust_success_rate=("count_robust_success", "mean"), mean_best_W=("best_W", "mean")).reset_index()
    per_task.to_csv(OUT / "per_task.csv", index=False)
    audit = {"protocol_sha256": hashlib.sha256(protocol_file.read_bytes()).hexdigest(),
        "policy_sha256": hashlib.sha256((ROOT / "scripts/adaptive_policy.py").read_bytes()).hexdigest(),
        "tasks": len(tasks), "repeats": protocol["repeats"], "methods": protocol["methods"],
        "queries": len(tasks) * protocol["repeats"] * len(protocol["methods"]) * 32,
        "heldout_double_labels_evaluated": 0, "wall_seconds": time.perf_counter() - start,
        "boundary": "Development exploration after inspecting pilot; no claim of independent formal improvement or Agnes performance."}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
