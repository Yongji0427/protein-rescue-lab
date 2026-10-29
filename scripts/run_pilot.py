"""Development-only GB1 data audit and label-budgeted baseline replay."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import beta, hypergeom


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "gb1_pilot_v1"


def canonical_pair(damage, rescue):
    return ":".join(sorted((damage, rescue), key=lambda x: int(x[1:-1])))


class QueryOracle:
    """Evaluator-owned labels; policies receive candidate IDs and single scores only.

    This is an API boundary for this pilot, not an OS sandbox for a future agent.
    """
    def __init__(self, damage, labels, budget):
        self.damage = damage
        self._labels = labels
        self.budget = budget
        self.cache = {}

    @property
    def used(self):
        return len(self.cache)

    def query(self, rescue):
        if rescue not in self._labels:
            raise ValueError("Candidate missing or violates fixed-damage constraint")
        if rescue in self.cache:
            return self.cache[rescue]
        if self.used >= self.budget:
            raise RuntimeError("Query budget exhausted")
        self.cache[rescue] = self._labels[rescue]
        return self.cache[rescue]


def count_interval(frame):
    """Exact conditional Poisson count-ratio sensitivity interval (90%).

    Assumes independent Poisson input/selection counts and fixed WT normalization.
    Does not capture biological variance or reproduce the authors' confidence filter.
    """
    selected = frame["Selection Count"].to_numpy(float)
    inputs = frame["Input Count"].to_numpy(float)
    lo_p = np.zeros(len(frame))
    positive = selected > 0
    lo_p[positive] = beta.ppf(.05, selected[positive], inputs[positive] + 1)
    hi_p = beta.ppf(.95, selected + 1, inputs)
    return lo_p / (1 - lo_p), hi_p / (1 - hi_p)


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    start = time.perf_counter()
    protocol_file = OUT / "protocol.json"
    protocol = json.loads(protocol_file.read_text())
    held = set(protocol["holdout_full_sequence_positions"])
    raw = pd.read_csv(ROOT / "data/evaluator_only/gb1_raw.csv")
    metadata = pd.read_csv(ROOT / "data/provenance/DMS_substitutions.csv")
    reference = metadata.loc[metadata.DMS_id == protocol["dataset"]].iloc[0]
    sequence = reference.target_seq
    is_single = raw["Mut2 Position"].isna()
    raw["p1"] = raw["Mut1 Position"] + 226
    raw["p2"] = raw["Mut2 Position"] + 226
    p1 = raw.p1.to_numpy(int)
    p2 = raw.p2.fillna(0).to_numpy(int)
    aa = np.array(list(sequence))
    assert len(raw) == reference.DMS_total_number_mutants
    assert is_single.sum() == 1045
    assert set(p1) == set(range(228, 283))
    assert (raw["Mut1 WT amino acid"].to_numpy() == aa[p1 - 1]).all()
    assert (raw.loc[~is_single, "Mut2 WT amino acid"].to_numpy() == aa[p2[~is_single] - 1]).all()
    assert (p1[~is_single] != p2[~is_single]).all()
    ids1 = raw["Mut1 WT amino acid"] + raw.p1.astype(str) + raw["Mut1 Mutation"]
    ids2 = raw["Mut2 WT amino acid"].fillna("") + raw.p2.fillna(0).astype(int).astype(str) + raw["Mut2 Mutation"].fillna("")
    assert (np.where(is_single, ids1, ids1 + ":" + ids2) == raw.mutant).all()
    raw["source_mutant"] = raw.mutant
    unordered_pairs = int(((p1 > p2) & ~is_single).sum())
    raw["mutant"] = np.where(is_single, ids1, np.where(p1 < p2, ids1 + ":" + ids2, ids2 + ":" + ids1))
    assert not raw.mutant.duplicated().any()

    # Before any score-based double-mutant analysis, remove every reserved position.
    dev = raw.loc[~is_single & ~raw.p1.isin(held) & ~raw.p2.isin(held)].copy()
    assert not set(dev.p1).intersection(held) and not set(dev.p2).intersection(held)
    singles = raw.loc[is_single].copy().set_index("mutant", drop=False)
    assert len(singles) == len(set(singles.p1)) * 19
    assert np.isfinite(singles.lnW).all() and np.isfinite(dev.lnW).all()
    assert (singles["Input Count"] > 0).all() and (dev["Input Count"] > 0).all()
    assert np.allclose(singles.lnW, np.log(singles["W_0.01floor"]), atol=1e-10)
    assert np.allclose(dev.lnW, np.log(dev["W_0.01floor"]), atol=1e-10)
    assert np.allclose(dev.W, dev["Selection Count"] / dev["Input Count"], atol=1e-10)
    assert np.allclose(singles.W, singles["Selection Count"] / singles["Input Count"], atol=1e-10)
    singles["count_ci_low"], singles["count_ci_high"] = count_interval(singles)
    dev["count_ci_low"], dev["count_ci_high"] = count_interval(dev)
    singles["W_display"] = np.exp(singles.lnW)
    dev["W_display"] = np.exp(dev.lnW)
    dev["damage1"] = ids1.loc[dev.index]
    dev["damage2"] = ids2.loc[dev.index]

    # Compare official processed scores only for development combinations and public singles.
    lookup = pd.concat([singles.set_index("mutant").lnW, dev.set_index("mutant").lnW])
    compared, sequence_checks = 0, 0
    max_score_error = 0.0
    for chunk in pd.read_csv(ROOT / "data/evaluator_only/gb1_processed.csv", chunksize=50000):
        chunk["mutant"] = chunk.mutant.map(lambda value: ":".join(sorted(value.split(":"), key=lambda x: int(x[1:-1]))))
        visible = chunk.loc[chunk.mutant.isin(lookup.index)]
        errors = np.abs(visible.DMS_score.to_numpy() - visible.mutant.map(lookup).to_numpy())
        max_score_error = max(max_score_error, float(errors.max(initial=0)))
        compared += len(visible)
        if sequence_checks < 1000:
            for row in visible.head(1000 - sequence_checks).itertuples():
                expected = list(sequence)
                for substitution in row.mutant.split(":"):
                    expected[int(substitution[1:-1]) - 1] = substitution[-1]
                assert row.mutated_sequence == "".join(expected)
                sequence_checks += 1
    assert compared == len(lookup) and max_score_error < 1e-10
    del raw

    public = ROOT / "data" / "public_reference"
    public.mkdir(parents=True, exist_ok=True)
    singles[["mutant", "p1", "lnW", "W", "W_display", "count_ci_low", "count_ci_high"]].to_csv(public / "singles.csv", index=False)
    pd.DataFrame({
        "full_position": range(228, 283), "paper_domain_position": range(2, 57),
        "measured_region_index": range(1, 56), "reference_aa": list(sequence[227:282]),
        "split": ["holdout" if p in held else "development" for p in range(228, 283)],
    }).to_csv(public / "position_mapping.csv", index=False)

    damage_rows = singles.loc[singles.W_display <= protocol["damage_relative_binding_max"]]
    damage_dev = damage_rows.loc[~damage_rows.p1.isin(held)]
    # Candidate identities in this manifest are public; held-out double scores are absent.
    holdout_manifest = damage_rows.loc[damage_rows.p1.isin(held), ["mutant", "p1", "lnW", "W_display"]]
    holdout_manifest.to_csv(OUT / "holdout_single_manifest.csv", index=False)
    groups1 = {key: group for key, group in dev.groupby("damage1", sort=False)}
    groups2 = {key: group for key, group in dev.groupby("damage2", sort=False)}
    empty = dev.iloc[0:0]
    task_records, excluded, run_records, curve_records, examples = [], [], [], [], []
    min_candidates = protocol["candidate_min_count"]
    budget = max(protocol["budgets"])
    seed = protocol["random_seed"]
    curve = {"random": np.zeros((protocol["random_repeats"], budget)), "wt_single_rank": np.zeros((1, budget))}
    curve_success = {key: np.zeros_like(value) for key, value in curve.items()}
    fields = ["damage", "method", "repeat", "query", "batch_cost", "rescue", "combination", "lnW", "W_display", "count_ci_low", "wt_rescue_lnW", "log_epistasis_floored"]
    with gzip.open(OUT / "query_log.csv.gz", "wt", newline="") as log:
        writer = csv.DictWriter(log, fieldnames=fields)
        writer.writeheader()
        for damage in damage_dev.itertuples():
            table = pd.concat([groups1.get(damage.mutant, empty), groups2.get(damage.mutant, empty)]).copy()
            table["rescue"] = np.where(table.damage1 == damage.mutant, table.damage2, table.damage1)
            table = table.loc[table.rescue.isin(singles.index)].sort_values("rescue").reset_index(drop=True)
            if len(table) < min_candidates:
                excluded.append({"damage": damage.mutant, "candidates": len(table), "reason": "coverage_below_500"})
                continue
            table["wt_lnW"] = table.rescue.map(singles.lnW)
            candidates = table[["rescue", "wt_lnW"]].copy()
            n = len(table)
            rescue_count = int((table.W_display >= .5).sum())
            robust_count = int((table.count_ci_low >= .5).sum())
            task_records.append({
                "damage": damage.mutant, "full_position": int(damage.p1), "domain_position": int(damage.p1 - 226),
                "damage_W_display": damage.W_display, "damage_W_raw": damage.W,
                "damage_count_ci_high": damage.count_ci_high, "damage_at_floor": bool(damage.W <= .01),
                "damage_ci_confirms_20pct": bool(damage.count_ci_high <= .2),
                "candidates": n, "rescue_candidates": rescue_count, "robust_rescue_candidates": robust_count,
                "strong_rescue_candidates": int((table.W_display >= 1).sum()),
                "best_available_W": float(table.W_display.max()),
                "rescue_fraction": rescue_count / n,
                **{f"random_exact_success_{b}": float(hypergeom.sf(0, n, rescue_count, b)) for b in protocol["budgets"]},
            })
            labels = {row.rescue: row for row in table.itertuples()}
            orders = [("wt_single_rank", 0, candidates.sort_values(["wt_lnW", "rescue"], ascending=[False, True]).rescue.head(budget).tolist())]
            for repeat in range(protocol["random_repeats"]):
                rng = np.random.default_rng(np.random.SeedSequence([seed, int(damage.p1), ord(damage.mutant[-1]), repeat]))
                orders.append(("random", repeat, rng.choice(candidates.rescue.to_numpy(), size=budget, replace=False).tolist()))
            for method, repeat, order in orders:
                oracle = QueryOracle(damage.mutant, labels, budget)
                best, robust_best = damage.W_display, 0.0
                first = None
                for q, rescue in enumerate(order, 1):
                    # Policy chose an ID using public singles, then requests its hidden combination label.
                    row = oracle.query(rescue)
                    assert oracle.used == q and int(rescue[1:-1]) != int(damage.p1)
                    assert canonical_pair(damage.mutant, rescue) == row.mutant
                    best = max(best, row.W_display)
                    robust_best = max(robust_best, row.count_ci_low)
                    cost = math.ceil(q / protocol["batch_size"]) * protocol["batch_size"]
                    if first is None and row.W_display >= .5:
                        first = cost
                    writer.writerow(dict(zip(fields, [damage.mutant, method, repeat, q, cost, rescue, row.mutant, row.lnW, row.W_display, row.count_ci_low, row.wt_lnW, row.lnW - damage.lnW - row.wt_lnW])))
                    curve[method][repeat, q - 1] += best
                    curve_success[method][repeat, q - 1] += best >= .5
                    if q in protocol["budgets"]:
                        run_records.append({"damage": damage.mutant, "method": method, "repeat": repeat, "budget": q, "best_W": best, "success": best >= .5, "strong_success": best >= 1.0, "count_robust_success": robust_best >= .5, "first_success_batch_cost": first, "queries_used": oracle.used, "reachable": rescue_count > 0,
                            "success_at_025": best >= .25, "success_at_100": best >= 1.0})
                if method == "wt_single_rank":
                    best_row = max(oracle.cache.values(), key=lambda x: x.W_display)
                    r_single = singles.loc[best_row.rescue]
                    examples.append({"damage": damage.mutant, "rescue": best_row.rescue, "combination": best_row.mutant,
                        "WT_W": 1.0, "damage_W_raw": damage.W, "damage_W_display": damage.W_display,
                        "rescue_single_W": float(r_single.W), "combination_W": best_row.W_display,
                        "combination_count_ci_low": best_row.count_ci_low, "combination_count_ci_high": best_row.count_ci_high,
                        "damage_count_ci_high": damage.count_ci_high, "damage_at_floor": bool(damage.W <= .01),
                        "log_epistasis_floored": best_row.lnW - damage.lnW - r_single.lnW,
                        "success": bool(best_row.W_display >= .5), "first_success_batch_cost": first})

    tasks = pd.DataFrame(task_records)
    runs = pd.DataFrame(run_records)
    tasks.to_csv(OUT / "development_tasks.csv", index=False)
    runs.to_csv(OUT / "runs.csv", index=False)
    pd.DataFrame(examples).to_csv(OUT / "wt_examples.csv", index=False)
    write_json("excluded_tasks.json", excluded)
    n_tasks = len(tasks)
    for method in curve:
        for q in range(1, budget + 1):
            curve_records.append({"method": method, "query": q, "mean_best_W": float(curve[method][:, q - 1].mean() / n_tasks),
                "success_rate": float(curve_success[method][:, q - 1].mean() / n_tasks),
                "random_exact_success": float(np.mean(hypergeom.sf(0, tasks.candidates, tasks.rescue_candidates, q)))})
    pd.DataFrame(curve_records).to_csv(OUT / "curves.csv", index=False)
    summary = runs.groupby(["method", "budget"], sort=True).agg(
        success_rate=("success", "mean"), strong_success_rate=("strong_success", "mean"),
        count_robust_success_rate=("count_robust_success", "mean"), mean_best_W=("best_W", "mean"),
        median_best_W=("best_W", "median"), success_at_025=("success_at_025", "mean"),
    ).reset_index()
    summary["task_count"] = n_tasks
    summary["repeats_per_task"] = summary.method.map({"random": protocol["random_repeats"], "wt_single_rank": 1})
    summary.to_csv(OUT / "summary.csv", index=False)
    write_json("audit.json", {
        "protocol_sha256": hashlib.sha256(protocol_file.read_bytes()).hexdigest(),
        "raw_records": int(reference.DMS_total_number_mutants), "single_records": len(singles),
        "development_double_records": len(dev), "damaging_singles_total": len(damage_rows),
        "development_damage_before_coverage": len(damage_dev), "development_tasks": n_tasks,
        "holdout_damaging_single_count": len(holdout_manifest), "coverage_exclusions": len(excluded),
        "reachable_development_tasks": int((tasks.rescue_candidates > 0).sum()),
        "unreachable_development_tasks": int((tasks.rescue_candidates == 0).sum()),
        "robust_reachable_tasks": int((tasks.robust_rescue_candidates > 0).sum()),
        "floor_damage_tasks": int(tasks.damage_at_floor.sum()),
        "damage_ci_confirms_20pct": int(tasks.damage_ci_confirms_20pct.sum()),
        "candidate_min": int(tasks.candidates.min()), "candidate_max": int(tasks.candidates.max()),
        "processed_scores_compared": compared, "processed_score_max_error": max_score_error,
        "processed_sequences_checked": sequence_checks,
        "source_pairs_requiring_order_normalization": unordered_pairs,
        "heldout_double_labels_evaluated": 0,
        "quality_note": "90% count-ratio intervals are a sampling sensitivity analysis with fixed normalization; not biological replicates or the original authors' high-confidence classification.",
        "isolation_note": "Strategies receive public candidate IDs/single scores only. Raw files remain evaluator-owned by convention; OS-level agent isolation is not implemented in this pilot.",
        "wall_seconds": time.perf_counter() - start,
    })
    print(json.dumps(json.loads((OUT / "audit.json").read_text()), ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
